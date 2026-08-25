"""Serve a pinned Qwen checkpoint through a minimal OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def _json_log(event: str, **values: Any) -> None:
    payload = {
        "timestamp_unix_s": time.time(),
        "event": event,
        **values,
    }
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


def _split_reasoning(text: str) -> tuple[str, str | None]:
    end_marker = "</think>"
    if end_marker not in text:
        return text.strip(), None
    reasoning, content = text.split(end_marker, maxsplit=1)
    return content.strip(), reasoning.removeprefix("<think>").strip()


class TransformersQwenEngine:
    """Single-GPU serial Qwen inference with plotting-ready request telemetry."""

    def __init__(self, model_id: str, revision: str) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the Transformers teacher server")
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        torch.set_float32_matmul_precision("high")
        _json_log(
            "model_load_started",
            model_id=model_id,
            revision=revision,
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),
        )
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model = model.to("cuda").eval()
        _json_log(
            "model_load_completed",
            model_id=model_id,
            revision=revision,
            latency_s=time.perf_counter() - started,
            memory_allocated_mib=torch.cuda.memory_allocated() / 2**20,
            memory_reserved_mib=torch.cuda.memory_reserved() / 2**20,
        )

    @staticmethod
    def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
        raw_messages = body.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a non-empty list")
        messages: list[dict[str, Any]] = []
        for raw in raw_messages:
            if not isinstance(raw, dict):
                raise ValueError("each message must be an object")
            role = str(raw.get("role", "user"))
            content = raw.get("content", "")
            if isinstance(content, str):
                normalized_content: Any = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_content = content
            else:
                raise ValueError("message content must be text or a content list")
            messages.append({"role": role, "content": normalized_content})
        return messages

    def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        messages = self._messages(body)
        seed = int(body.get("seed", 0))
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 1.0))
        top_k = int(body.get("top_k", 0))
        max_tokens = int(body.get("max_tokens", 2048))
        presence_penalty = float(body.get("presence_penalty", 0.0))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to("cuda")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generation: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": True,
        }
        if temperature > 0.0:
            generation.update({"temperature": temperature, "top_p": top_p})
            if top_k > 0:
                generation["top_k"] = top_k
        if presence_penalty != 0.0:
            from transformers import LogitsProcessorList

            class PresencePenaltyProcessor:
                def __call__(self, input_ids: Any, scores: Any) -> Any:
                    adjusted = scores.clone()
                    for batch_index in range(input_ids.shape[0]):
                        seen_tokens = torch.unique(input_ids[batch_index])
                        adjusted[batch_index, seen_tokens] -= presence_penalty
                    return adjusted

            generation["logits_processor"] = LogitsProcessorList([PresencePenaltyProcessor()])

        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        latency_s = time.perf_counter() - started
        generated_ids = output[0, prompt_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        decoded = self.processor.decode(generated_ids, skip_special_tokens=True)
        content, reasoning = _split_reasoning(decoded)
        request_id = str(body.get("user") or uuid.uuid4().hex)
        _json_log(
            "generation_completed",
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            tokens_per_second=(completion_tokens / latency_s if latency_s > 0.0 else 0.0),
            peak_memory_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_memory_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": "length" if completion_tokens >= max_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class TeacherRequestHandler(BaseHTTPRequestHandler):
    engine: TransformersQwenEngine

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": self.engine.model_id})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            self._send_json(200, self.engine.generate(body))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _json_log(
                "generation_failed",
                request_id=str(body.get("user", "unknown")) if "body" in locals() else "unknown",
                error_type=type(error).__name__,
                error=str(error),
            )
            self._send_json(
                500,
                {"error": {"type": type(error).__name__, "message": str(error)}},
            )

    def log_message(self, format_string: str, *arguments: Any) -> None:
        _json_log(
            "http_request",
            client=self.client_address[0],
            message=format_string % arguments,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    engine = TransformersQwenEngine(arguments.model, arguments.revision)
    TeacherRequestHandler.engine = engine
    server = HTTPServer((arguments.host, arguments.port), TeacherRequestHandler)
    _json_log(
        "server_ready",
        host=arguments.host,
        port=arguments.port,
        model_id=arguments.model,
        revision=arguments.revision,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        _json_log("server_stopped")


if __name__ == "__main__":
    main()
