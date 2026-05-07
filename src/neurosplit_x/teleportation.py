"""Neural Teleportation Layer (NTL)."""

from __future__ import annotations

import hashlib

from .models import NetworkState, TeleportationPlan


class NeuralTeleportationOptimizer:
    """Plans predictive layer migration with checkpoint continuity."""

    def plan(
        self,
        source: str,
        destination: str,
        split_layer: int,
        model_layers: int,
        network: NetworkState,
    ) -> TeleportationPlan:
        if split_layer >= model_layers:
            raise ValueError("split_layer must be less than model_layers")
        future_layers = list(range(split_layer + 1, min(model_layers, split_layer + 4) + 1))
        compression_ratio = max(0.10, min(0.88, 0.72 - network.packet_loss * 1.8 - network.jitter_ms / 250.0))
        continuity_window_ms = network.latency_ms + network.jitter_ms + 12.0
        token_seed = f"{source}->{destination}:{future_layers}:{compression_ratio:.3f}"
        checkpoint_token = hashlib.blake2b(token_seed.encode(), digest_size=16).hexdigest()
        clone_count = 1 if network.bandwidth_mbps < 50 else 2 if network.bandwidth_mbps < 200 else 3
        return TeleportationPlan(
            source=source,
            destination=destination,
            layers=future_layers,
            checkpoint_token=checkpoint_token,
            compression_ratio=compression_ratio,
            continuity_window_ms=continuity_window_ms,
            clone_count=clone_count,
        )
