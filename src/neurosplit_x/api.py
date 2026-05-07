"""FastAPI control plane for NeuroSplit-X."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .digital_twin import DigitalTwinUniverse
from .models import DeviceState, NetworkState, SplitPlan, WorkloadState
from .splitting import CognitiveNeuralSplittingEngine
from .teleportation import NeuralTeleportationOptimizer

app = FastAPI(
    title="NeuroSplit-X Control Plane",
    version="0.1.0",
    description="Autonomous distributed edge-cloud cognitive intelligence API.",
)
engine = CognitiveNeuralSplittingEngine()
twin = DigitalTwinUniverse()
teleporter = NeuralTeleportationOptimizer()


class PlanRequest(BaseModel):
    request_id: str
    model_layers: int = Field(ge=2)
    device: DeviceState
    network: NetworkState
    workload: WorkloadState


class PlanResponse(BaseModel):
    split_plan: SplitPlan
    teleportation: dict[str, object]
    digital_twin_summary: dict[str, float]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "system": "neurosplit-x"}


@app.post("/v1/infer/plan", response_model=PlanResponse)
def plan_inference(payload: PlanRequest) -> PlanResponse:
    split_plan = engine.plan(
        payload.request_id,
        payload.model_layers,
        payload.device,
        payload.network,
        payload.workload,
    )
    scenarios = twin.simulate(payload.device, payload.network, samples=32)
    teleportation = teleporter.plan(
        source=payload.device.id,
        destination="cloud-primary",
        split_layer=split_plan.split_layer,
        model_layers=payload.model_layers,
        network=payload.network,
    )
    return PlanResponse(
        split_plan=split_plan,
        teleportation=asdict(teleportation),
        digital_twin_summary=twin.summarize(scenarios),
    )
