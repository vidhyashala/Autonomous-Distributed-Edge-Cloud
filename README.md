# NeuroSplit-X: Autonomous Distributed Edge-Cloud Cognitive Intelligence Framework

NeuroSplit-X is an invention-grade research and reference implementation for a living distributed neural organism: a cognitive edge-cloud AI fabric where neural networks dynamically split, migrate, compress, replicate, and self-evolve across edge devices, swarms, and cloud infrastructure.

The repository contains:

- A full research architecture and patent-oriented specification in `docs/neurosplit_x_framework.md`.
- A runnable FastAPI control plane in `src/neurosplit_x/api.py`.
- Core algorithmic implementations for predictive splitting, teleportation, swarm routing, digital twins, and meta-orchestration in `src/neurosplit_x/`.
- Kubernetes, Docker, Rust, Go, and React/TypeScript scaffolds for a production edge-cloud deployment.
- Unit tests covering the deterministic reference algorithms.

> Status: reference research framework and executable prototype scaffolding. Large-scale training against ImageNet, Google Cluster Traces, Waymo, The Pile, and other datasets is specified as a reproducible pipeline but intentionally not bundled in this repository.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
uvicorn neurosplit_x.api:app --reload
```

## Example API

```bash
curl -X POST http://127.0.0.1:8000/v1/infer/plan \
  -H 'content-type: application/json' \
  -d '{"request_id":"demo","model_layers":12,"device":{"id":"edge-1","battery":0.82,"compute_tflops":2.1,"thermal_c":46},"network":{"latency_ms":18,"bandwidth_mbps":90,"jitter_ms":3},"workload":{"semantic_class":"vision","urgency":0.7,"privacy":0.8}}'
```

## Architecture at a Glance

NeuroSplit-X combines five novel invention modules:

1. **Cognitive Neural Splitting Engine (CNSE)** — predicts and mutates split points before inference starts.
2. **Neural Teleportation Layer (NTL)** — live-migrates neural layers, tensor states, and compressed checkpoints.
3. **Digital Twin Edge-Cloud Universe (DTECU)** — simulates future infrastructure states before deployment decisions.
4. **Swarm Edge Intelligence Network (SEIN)** — lets nearby edge devices cooperate on partial inference.
5. **Self-Evolving Meta-AI Orchestrator (SEMO)** — evolves policies, architectures, and orchestration rules.

## Repository Layout

```text
src/neurosplit_x/             Python research prototype and FastAPI backend
rust/neurosplit_opt/          Rust optimization-engine skeleton
agents/go-orchestrator/       Go orchestration-agent skeleton
frontend/neurosplit-dashboard React + TypeScript dashboard scaffold
k8s/                          Kubernetes deployment manifests
docker/                       Docker Compose integration stack
docs/                         Research, patent, and system design documentation
tests/                        Pytest suite
```
