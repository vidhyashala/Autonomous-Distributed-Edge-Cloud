from neurosplit_x.digital_twin import DigitalTwinUniverse
from neurosplit_x.models import DeviceState, NetworkState, SwarmPeer, WorkloadState
from neurosplit_x.splitting import CognitiveNeuralSplittingEngine
from neurosplit_x.swarm import SwarmInferenceCoordinator
from neurosplit_x.teleportation import NeuralTeleportationOptimizer


def sample_context(packet_loss: float = 0.0):
    return (
        DeviceState(id="edge-1", battery=0.82, compute_tflops=2.1, thermal_c=46.0, trusted_enclave=True),
        NetworkState(latency_ms=18.0, bandwidth_mbps=90.0, jitter_ms=3.0, packet_loss=packet_loss),
        WorkloadState(semantic_class="vision", urgency=0.7, privacy=0.8, tensor_mb=12.0),
    )


def test_split_plan_is_valid_and_audited():
    device, network, workload = sample_context()
    plan = CognitiveNeuralSplittingEngine().plan("req-1", 12, device, network, workload)

    assert 1 <= plan.split_layer < 12
    assert plan.expected_latency_ms > 0
    assert 0.0 <= plan.split_entropy <= 1.0
    assert len(plan.audit_hash) == 64
    assert plan.explanation


def test_packet_loss_triggers_more_migration_pressure():
    device, stable, workload = sample_context(packet_loss=0.0)
    _, lossy, _ = sample_context(packet_loss=0.2)

    stable_plan = CognitiveNeuralSplittingEngine().plan("stable", 12, device, stable, workload)
    lossy_plan = CognitiveNeuralSplittingEngine().plan("lossy", 12, device, lossy, workload)

    assert lossy_plan.compression_ratio < stable_plan.compression_ratio


def test_teleportation_generates_future_layers():
    _, network, _ = sample_context()
    plan = NeuralTeleportationOptimizer().plan("edge-1", "cloud-1", 5, 12, network)

    assert plan.layers == [6, 7, 8, 9]
    assert plan.clone_count >= 1
    assert plan.checkpoint_token


def test_digital_twin_summary_contains_probabilities():
    device, network, _ = sample_context()
    scenarios = DigitalTwinUniverse(seed=1).simulate(device, network, samples=8)
    summary = DigitalTwinUniverse.summarize(scenarios)

    assert 0.0 <= summary["collapse_probability_mean"] <= 1.0
    assert summary["latency_pseudo_mean_ms"] > 0.0


def test_swarm_filters_low_trust_peers():
    peers = [
        SwarmPeer(id="bad", distance_m=1, trust_score=0.2, available_tflops=10, bandwidth_mbps=100),
        SwarmPeer(id="good", distance_m=3, trust_score=0.9, available_tflops=1, bandwidth_mbps=80),
    ]

    selected = SwarmInferenceCoordinator().choose_peers(peers, tensor_mb=8)

    assert [peer.id for peer in selected] == ["good"]
