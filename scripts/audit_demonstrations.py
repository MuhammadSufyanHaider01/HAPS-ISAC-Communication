"""Audit a generated teacher dataset before scaling or training."""

from __future__ import annotations

import argparse
import json

from haps_isac.data.audit import audit_dataset
from haps_isac.data.demonstration_schema import json_safe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--fail-on-warning", action="store_true")
    arguments = parser.parse_args()
    audit = audit_dataset(arguments.dataset)
    print(json.dumps(json_safe(audit), indent=2, sort_keys=True))
    if not audit.passed or (arguments.fail_on_warning and audit.warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
