"""Cognitive Neural Splitting Engine (CNSE)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .models import DeviceState, NetworkState, Placement, SplitPlan, WorkloadState


@dataclass(frozen=True)
class ObjectiveWeights:
    latency: float = 0.38
    energy: float = 0.24
    bandwidth: float = 0.16
    privacy: float = 0.14
    stability: float = 0.08


class CognitiveNeuralSplittingEngine:
    """Reference implementation of Predictive Cognitive Split Routing.

    Production NeuroSplit-X trains this policy with graph neural encoders and
    reinforcement learning. This deterministic implementation mirrors the
    objective so it can be tested, audited, and used as a safe fallback policy.
    """

    def __init__(self, weights: ObjectiveWeights | None = None) -> None:
        self.weights = weights or ObjectiveWeights()

    def plan(
        self,
        request_id: str,
        model_layers: int,
        device: DeviceState,
        network: NetworkState,
        workload: WorkloadState,
    ) -> SplitPlan:
        """Choose a split layer and placement before inference starts."""

        if model_layers < 2:
            raise ValueError("model_layers must be at least 2")

        candidates = [self._score_layer(layer, model_layers, device, network, workload) for layer in range(1, model_layers)]
        best = min(candidates, key=lambda row: row["score"])
        placement = self._placement(best["edge_fraction"], workload, network)
        audit_material = f"{request_id}:{best['layer']}:{placement}:{best['latency']:.4f}:{best['energy']:.4f}"
        audit_hash = hashlib.sha256(audit_material.encode()).hexdigest()
        explanation = [
            f"Selected split {best['layer']} because predicted score {best['score']:.3f} is minimal.",
            f"Split entropy {best['entropy']:.3f} balances edge compute, network volatility, and privacy pressure.",
            f"Placement {placement.value} reflects edge fraction {best['edge_fraction']:.2f} and urgency {workload.urgency:.2f}.",
        ]
        return SplitPlan(
            request_id=request_id,
            split_layer=int(best["layer"]),
            placement=placement,
            split_entropy=float(best["entropy"]),
            expected_latency_ms=float(best["latency"]),
            expected_energy_j=float(best["energy"]),
            compression_ratio=float(best["compression"]),
            migration_required=bool(best["mutation_pressure"] > 0.55),
            explanation=explanation,
            audit_hash=audit_hash,
        )

    def _score_layer(
        self,
        layer: int,
        model_layers: int,
        device: DeviceState,
        network: NetworkState,
        workload: WorkloadState,
    ) -> dict[str, float]:
        edge_fraction = layer / model_layers
        cloud_fraction = 1.0 - edge_fraction
        compute_latency = 9.5 * edge_fraction / math.log2(device.compute_tflops + 2.0)
        transfer_latency = workload.tensor_mb * cloud_fraction * 8.0 / network.bandwidth_mbps * 1000.0
        volatility = network.jitter_ms + 120.0 * network.packet_loss + 2.5 * network.mobility_speed_mps
        privacy_penalty = workload.privacy * cloud_fraction * 30.0
        thermal_penalty = device.thermal_pressure * edge_fraction * 25.0
        latency = network.latency_ms + compute_latency + transfer_latency + volatility + thermal_penalty
        energy = 0.45 + 2.2 * edge_fraction * (1.0 + device.thermal_pressure) + 0.7 * cloud_fraction
        compression = max(0.08, min(0.95, 0.9 - 0.52 * network.packet_loss - 0.24 * workload.urgency))
        entropy = self._split_entropy(edge_fraction, network, workload)
        mutation_pressure = min(1.0, entropy + device.thermal_pressure * 0.3 + network.packet_loss * 0.8)
        bandwidth_cost = workload.tensor_mb * cloud_fraction * compression
        score = (
            self.weights.latency * latency
            + self.weights.energy * energy * 10.0
            + self.weights.bandwidth * bandwidth_cost
            + self.weights.privacy * privacy_penalty
            + self.weights.stability * mutation_pressure * 20.0
        )
        return {
            "layer": float(layer),
            "score": score,
            "latency": latency,
            "energy": energy,
            "compression": compression,
            "entropy": entropy,
            "mutation_pressure": mutation_pressure,
            "edge_fraction": edge_fraction,
        }

    @staticmethod
    def _split_entropy(edge_fraction: float, network: NetworkState, workload: WorkloadState) -> float:
        p_edge = max(1e-6, min(1.0 - 1e-6, edge_fraction))
        binary_entropy = -(p_edge * math.log2(p_edge) + (1.0 - p_edge) * math.log2(1.0 - p_edge))
        context_pressure = 0.35 * workload.urgency + 0.25 * workload.privacy + 0.4 * min(1.0, network.jitter_ms / 50.0)
        return max(0.0, min(1.0, 0.65 * binary_entropy + 0.35 * context_pressure))

    @staticmethod
    def _placement(edge_fraction: float, workload: WorkloadState, network: NetworkState) -> Placement:
        if workload.privacy > 0.82 and edge_fraction > 0.55:
            return Placement.EDGE
        if network.packet_loss > 0.08 or network.jitter_ms > 40.0:
            return Placement.SWARM
        if edge_fraction < 0.25 and workload.urgency < 0.5:
            return Placement.CLOUD
        return Placement.HYBRID
