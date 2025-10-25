from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DemandRequest(BaseModel):
    date: date
    plant_id: int = Field(..., ge=1)
    base_quantity: Optional[float] = Field(
        None,
        ge=0,
        description="Cantidad base (spec) para ajustar con el IDD.",
    )
    historical_avg_passengers: Optional[float] = Field(
        None,
        ge=0,
        description="Promedio historico de pasajeros. Si no se provee se usa el baseline del modelo.",
    )


class DemandResponse(BaseModel):
    date: date
    plant_id: int
    predicted_passengers: Optional[float]
    baseline_passengers: Optional[float]
    demand_index: Optional[float] = Field(
        None,
        description="IDD de pasajeros (pred / baseline).",
    )
    recommended_quantity: Optional[int] = Field(
        None,
        description="Cantidad recomendada (redondeada hacia arriba) si se proporciono base_quantity.",
    )


class StaffingRequest(BaseModel):
    date: date
    plant_id: int = Field(..., ge=1)
    staff_baseline: Optional[float] = Field(
        None,
        ge=0,
        description="Numero base de empleados para ajustar segun el IDD.",
    )
    historical_avg_flights: Optional[float] = Field(
        None,
        ge=0,
        description="Promedio historico de vuelos. Si no se provee se usa el baseline del modelo.",
    )


class StaffingResponse(BaseModel):
    date: date
    plant_id: int
    predicted_flights: float
    baseline_flights: Optional[float]
    workload_index: Optional[float] = Field(
        None,
        description="IDD de vuelos (pred / baseline).",
    )
    recommended_staff: Optional[int] = Field(
        None,
        description="Personal recomendado (ceil) si se proporciono staff_baseline.",
    )


class GeminiPlantPayload(BaseModel):
    plant_id: int
    name: Optional[str] = None
    target_date: Optional[date] = None
    description: Optional[str] = None
    base_quantity: Optional[float] = None
    staff_baseline: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GeminiSimulationRequest(BaseModel):
    date: date
    scenario_id: Optional[str] = None
    plants: List[GeminiPlantPayload]
    gemini_metadata: Dict[str, Any] = Field(default_factory=dict)


class SimulationPlantResult(BaseModel):
    plant_id: int
    name: Optional[str]
    target_date: date
    gemini_prompt: str
    gemini_raw: Dict[str, Any]
    gemini_payload: Dict[str, Any]
    demand_prediction: Optional[DemandResponse]
    staffing_prediction: StaffingResponse


class GeminiSimulationResponse(BaseModel):
    date: date
    scenario_id: Optional[str]
    results: List[SimulationPlantResult]
    gemini_metadata: Dict[str, Any] = Field(default_factory=dict)
