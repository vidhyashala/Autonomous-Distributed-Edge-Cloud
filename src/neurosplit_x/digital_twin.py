"""Digital Twin Edge-Cloud Universe."""

from __future__ import annotations

import random
from statistics import mean

from .models import DeviceState, NetworkState, TwinScenario


class DigitalTwinUniverse:
    """Monte Carlo future-state simulator for predictive orchestration."""

    def __init__(self, seed: int = 7) -> None:
        self._random = random.Random(seed)

    def simulate(self, device: DeviceState, network: NetworkState, horizon_ms: int = 1000, samples: int = 64) -> list[TwinScenario]:
        scenarios: list[TwinScenario] = []
        for _ in range(samples):
            mobility_drift = network.mobility_speed_mps * self._random.uniform(0.05, 0.25)
            jitter_shock = self._random.expovariate(1.0 / max(1.0, network.jitter_ms + 1.0))
            collapse = min(0.98, network.packet_loss * 2.8 + mobility_drift / 20.0 + device.thermal_pressure * 0.25)
            latency = network.latency_ms + jitter_shock + mobility_drift * 7.5 + collapse * 80.0
            bandwidth = max(1.0, network.bandwidth_mbps * (1.0 - collapse * self._random.uniform(0.2, 0.75)))
            energy = 0.8 + device.thermal_pressure * 3.0 + latency / 200.0
            scenarios.append(
                TwinScenario(
                    horizon_ms=horizon_ms,
                    predicted_latency_ms=latency,
                    predicted_bandwidth_mbps=bandwidth,
                    predicted_energy_j=energy,
                    collapse_probability=collapse,
                )
            )
        return scenarios

    @staticmethod
    def summarize(scenarios: list[TwinScenario]) -> dict[str, float]:
        return {
            "latency_pseudo_mean_ms": mean(s.predicted_latency_ms for s in scenarios),
            "bandwidth_pseudo_mean_mbps": mean(s.predicted_bandwidth_mbps for s in scenarios),
            "energy_pseudo_mean_j": mean(s.predicted_energy_j for s in scenarios),
            "collapse_probability_mean": mean(s.collapse_probability for s in scenarios),
        }
