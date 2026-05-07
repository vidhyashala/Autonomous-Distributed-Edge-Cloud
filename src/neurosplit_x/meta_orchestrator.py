"""Self-Evolving Meta-AI Orchestrator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyGenome:
    """Compact policy genome evolved by SEMOA."""

    split_aggressiveness: float
    privacy_bias: float
    swarm_affinity: float
    teleportation_bias: float


class MetaOrchestrator:
    """Evolutionary policy generator for orchestration strategies."""

    def mutate(self, genome: PolicyGenome, reward: float) -> PolicyGenome:
        direction = 1.0 if reward >= 0.0 else -1.0
        step = min(0.08, 0.01 + abs(reward) / 100.0)
        return PolicyGenome(
            split_aggressiveness=self._clip(genome.split_aggressiveness + direction * step),
            privacy_bias=self._clip(genome.privacy_bias + direction * step * 0.7),
            swarm_affinity=self._clip(genome.swarm_affinity + direction * step * 0.5),
            teleportation_bias=self._clip(genome.teleportation_bias + direction * step * 0.6),
        )

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, value))
