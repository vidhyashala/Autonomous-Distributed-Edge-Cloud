"""Shared NeuroSplit-X data models.

Core models are standard-library dataclasses so deterministic planning can run on
minimal edge runtimes. The FastAPI adapter can still serialize them into API
responses through ``dataclasses.asdict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Placement(str, Enum):
    """Execution placement for a model segment."""

    EDGE = "edge"
    SWARM = "swarm"
    CLOUD = "cloud"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class DeviceState:
    """Telemetry for an edge or cloud execution node."""

    id: str
    battery: float
    compute_tflops: float
    thermal_c: float
    memory_gb: float = 4.0
    accelerator: str = "cpu"
    trusted_enclave: bool = False

    def __post_init__(self) -> None:
        _range("battery", self.battery, 0.0, 1.0)
        _positive("compute_tflops", self.compute_tflops)
        _positive("memory_gb", self.memory_gb)

    @property
    def thermal_pressure(self) -> float:
        """Normalized throttling pressure used by schedulers."""

        return max(0.0, min(1.0, (self.thermal_c - 45.0) / 45.0))


@dataclass(frozen=True)
class NetworkState:
    """Network observations and short-horizon measurements."""

    latency_ms: float
    bandwidth_mbps: float
    jitter_ms: float = 0.0
    packet_loss: float = 0.0
    mobility_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        _non_negative("latency_ms", self.latency_ms)
        _positive("bandwidth_mbps", self.bandwidth_mbps)
        _non_negative("jitter_ms", self.jitter_ms)
        _range("packet_loss", self.packet_loss, 0.0, 1.0)
        _non_negative("mobility_speed_mps", self.mobility_speed_mps)


@dataclass(frozen=True)
class WorkloadState:
    """Semantic and policy context for an inference request."""

    semantic_class: str
    urgency: float
    privacy: float
    tensor_mb: float = 12.0
    accuracy_floor: float = 0.90
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _range("urgency", self.urgency, 0.0, 1.0)
        _range("privacy", self.privacy, 0.0, 1.0)
        _positive("tensor_mb", self.tensor_mb)
        _range("accuracy_floor", self.accuracy_floor, 0.0, 1.0)


@dataclass(frozen=True)
class SplitPlan:
    """A complete adaptive inference plan."""

    request_id: str
    split_layer: int
    placement: Placement
    split_entropy: float
    expected_latency_ms: float
    expected_energy_j: float
    compression_ratio: float
    migration_required: bool
    explanation: list[str]
    audit_hash: str


@dataclass(frozen=True)
class TeleportationPlan:
    """Neural layer migration and replication plan."""

    source: str
    destination: str
    layers: list[int]
    checkpoint_token: str
    compression_ratio: float
    continuity_window_ms: float
    clone_count: int


@dataclass(frozen=True)
class SwarmPeer:
    """A candidate peer for collaborative inference."""

    id: str
    distance_m: float
    trust_score: float
    available_tflops: float
    bandwidth_mbps: float

    def __post_init__(self) -> None:
        _non_negative("distance_m", self.distance_m)
        _range("trust_score", self.trust_score, 0.0, 1.0)
        _non_negative("available_tflops", self.available_tflops)
        _positive("bandwidth_mbps", self.bandwidth_mbps)


@dataclass(frozen=True)
class TwinScenario:
    """Monte Carlo future-state sample for the digital twin."""

    horizon_ms: int
    predicted_latency_ms: float
    predicted_bandwidth_mbps: float
    predicted_energy_j: float
    collapse_probability: float


def _positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _range(name: str, value: float, lower: float, upper: float) -> None:
    if value < lower or value > upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
