"""Validated immutable experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    """Base class for strict immutable configuration groups."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureConfig(FrozenModel):
    noma: bool = True
    sensing: bool = True
    target_tracking: bool = True
    aerial_ris: bool = False
    aav_jamming: bool = False
    residual_self_interference: bool = False
    imperfect_csi: bool = False
    imperfect_sic: bool = False
    stochastic_blockage: bool = False


class SystemConfig(FrozenModel):
    num_noma_pairs: int = Field(gt=0)
    slot_duration_s: float = Field(gt=0.0)
    episode_slots: int = Field(gt=0)
    bandwidth_hz: float = Field(gt=0.0)
    carrier_frequency_hz: float = Field(gt=0.0)


class HapsConfig(FrozenModel):
    position_m: tuple[float, float, float]
    num_tx_antennas: int = Field(gt=0)
    num_rx_antennas: int = Field(gt=0)
    max_power_w: float = Field(gt=0.0)
    min_cpu_frequency_hz: float = Field(gt=0.0)
    max_cpu_frequency_hz: float = Field(gt=0.0)
    sensing_cycles: float = Field(gt=0.0)
    effective_capacitance: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_cpu_range(self) -> Self:
        if self.max_cpu_frequency_hz < self.min_cpu_frequency_hz:
            raise ValueError("max_cpu_frequency_hz must not be below the minimum")
        if self.position_m[2] <= 0.0:
            raise ValueError("the HAPS altitude must be positive")
        return self


class ChannelConfig(FrozenModel):
    rician_k_factor: float = Field(ge=0.0)
    additional_loss_db: float = Field(ge=0.0)
    atmospheric_loss_db: float = Field(ge=0.0)
    thermal_noise_density_dbm_hz: float = -174.0
    receiver_noise_figure_db: float = Field(ge=0.0)
    sensing_cancellation_fraction: float = Field(ge=0.0, le=1.0)
    target_sic_residual_fraction: float = Field(ge=0.0, le=1.0)
    legitimate_sic_residual_fraction: float = Field(ge=0.0, le=1.0)
    csi_error_variance: float = Field(ge=0.0)


class TrafficConfig(FrozenModel):
    packet_bits_near: float = Field(gt=0.0)
    packet_bits_far: float = Field(gt=0.0)


class SensingConfig(FrozenModel):
    echo_gain: float = Field(gt=0.0)
    sinr_threshold: float = Field(gt=0.0)
    sinr_floor: float = Field(gt=0.0)
    range_variance_scale: float = Field(gt=0.0)
    angle_variance_scale: float = Field(gt=0.0)
    doppler_variance_scale: float = Field(gt=0.0)
    acceptance_covariance_trace: float = Field(gt=0.0)
    residual_self_interference_fraction: float = Field(ge=0.0, le=1.0)


class TargetConfig(FrozenModel):
    initial_state: tuple[float, float, float, float]
    initial_std: tuple[float, float, float, float]
    acceleration_std_mps2: float = Field(ge=0.0)
    scenario_radius_m: float = Field(gt=0.0)
    min_target_radius_m: float = Field(gt=0.0)
    max_target_radius_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_target_radii(self) -> Self:
        if self.max_target_radius_m <= self.min_target_radius_m:
            raise ValueError("target maximum radius must exceed its minimum radius")
        if any(value <= 0.0 for value in self.initial_std):
            raise ValueError("initial target standard deviations must be positive")
        return self


class TopologyConfig(FrozenModel):
    near_radius_range_m: tuple[float, float]
    far_radius_range_m: tuple[float, float]
    minimum_pair_separation_m: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        for name, bounds in (
            ("near_radius_range_m", self.near_radius_range_m),
            ("far_radius_range_m", self.far_radius_range_m),
        ):
            if bounds[0] <= 0.0 or bounds[1] <= bounds[0]:
                raise ValueError(f"invalid {name}")
        if self.near_radius_range_m[1] >= self.far_radius_range_m[0]:
            raise ValueError("near and far radial ranges must not overlap")
        return self


class FreshnessConfig(FrozenModel):
    aoi_cap_slots: int = Field(gt=1)
    aoli_cap_slots: int = Field(gt=1)
    aosi_cap_slots: int = Field(gt=1)
    initial_aoi_slots: int = Field(ge=1)
    initial_aoli_slots: int = Field(ge=1)
    initial_aosi_slots: int = Field(ge=1)


class ConstraintConfig(FrozenModel):
    minimum_aoli_slots: float = Field(ge=0.0)
    minimum_delivery_rate: float = Field(ge=0.0, le=1.0)
    secrecy_outage_probability: float = Field(ge=0.0, le=1.0)
    sensing_outage_probability: float = Field(ge=0.0, le=1.0)
    secrecy_rate_target_bps: float = Field(ge=0.0)
    sensing_sinr_threshold: float = Field(gt=0.0)
    sic_sinr_threshold: float = Field(gt=0.0)
    maximum_covariance_trace: float = Field(gt=0.0)
    minimum_sensing_power_w: float = Field(ge=0.0)
    queue_reference: float = Field(gt=0.0)
    queue_clip: float = Field(gt=0.0)


class ObjectiveConfig(FrozenModel):
    weight_aoi: float = Field(ge=0.0)
    weight_aosi: float = Field(ge=0.0)
    weight_uncertainty: float = Field(ge=0.0)
    weight_energy: float = Field(ge=0.0)
    covariance_reference: float = Field(gt=0.0)
    energy_reference_j: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        total = (
            self.weight_aoi
            + self.weight_aosi
            + self.weight_uncertainty
            + self.weight_energy
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("objective weights must sum to one")
        return self


class EpisodeConfig(FrozenModel):
    warmup_slots: int = Field(ge=0)
    evaluation_slots: int = Field(gt=0)


class ExperimentConfig(FrozenModel):
    schema_version: int = Field(ge=1)
    experiment_name: str
    master_seed: int = Field(ge=0)
    features: FeatureConfig
    system: SystemConfig
    haps: HapsConfig
    channels: ChannelConfig
    topology: TopologyConfig
    traffic: TrafficConfig
    sensing: SensingConfig
    target: TargetConfig
    freshness: FreshnessConfig
    constraints: ConstraintConfig
    objective: ObjectiveConfig
    episode: EpisodeConfig

    @model_validator(mode="after")
    def validate_v1_contract(self) -> Self:
        if not (self.features.noma and self.features.sensing and self.features.target_tracking):
            raise ValueError("Version 1 requires NOMA, sensing, and target tracking")
        if self.constraints.minimum_sensing_power_w > self.haps.max_power_w:
            raise ValueError("minimum sensing power exceeds the HAPS power budget")
        if self.constraints.sensing_sinr_threshold != self.sensing.sinr_threshold:
            raise ValueError("sensing SINR thresholds must agree")
        return self

    @property
    def num_users(self) -> int:
        return 2 * self.system.num_noma_pairs

    @property
    def num_virtual_queues(self) -> int:
        return 6 * self.system.num_noma_pairs + 2


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and fully validate an experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return ExperimentConfig.model_validate(raw)
