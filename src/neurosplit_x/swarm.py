"""Swarm Edge Intelligence Network."""

from __future__ import annotations

from .models import SwarmPeer


class SwarmInferenceCoordinator:
    """Selects trusted nearby collaborators for partial inference."""

    def choose_peers(self, peers: list[SwarmPeer], tensor_mb: float, max_peers: int = 3) -> list[SwarmPeer]:
        ranked = sorted(peers, key=lambda peer: self._utility(peer, tensor_mb), reverse=True)
        return [peer for peer in ranked if peer.trust_score >= 0.55 and peer.available_tflops > 0.05][:max_peers]

    @staticmethod
    def _utility(peer: SwarmPeer, tensor_mb: float) -> float:
        communication_penalty = tensor_mb * 8.0 / peer.bandwidth_mbps + peer.distance_m / 100.0
        return peer.trust_score * peer.available_tflops / (1.0 + communication_penalty)
