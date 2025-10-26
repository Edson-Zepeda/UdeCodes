from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .financial import (
    DEFAULT_FUEL_COST_PER_LITER,
    DEFAULT_UNIT_MARGIN_FACTOR,
    FUEL_BURN_LITERS_PER_KG,
    RECOMMENDED_BUFFER,
)


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
    historical_avg_passengers: Optional[float] = Field(
        None, ge=0, description="Promedio historico de pasajeros para referencia."
    )
    historical_avg_flights: Optional[float] = Field(
        None, ge=0, description="Promedio historico de vuelos para referencia."
    )
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


class FinancialImpactRequest(BaseModel):
    fuel_cost_per_liter: float = Field(
        DEFAULT_FUEL_COST_PER_LITER,
        ge=0,
        description="Costo del combustible por litro utilizado para el escenario.",
    )
    fuel_burn_liters_per_kg: float = Field(
        FUEL_BURN_LITERS_PER_KG,
        ge=0,
        description="Litros de combustible ahorrados por cada kilogramo removido.",
    )
    buffer_factor: float = Field(
        RECOMMENDED_BUFFER,
        ge=0,
        description="Factor de seguridad aplicado a las predicciones del modelo.",
    )
    waste_cost_multiplier: float = Field(
        1.0,
        ge=0,
        description="Factor para escalar los costes de desperdicio segun la entrada del usuario.",
    )
    unit_margin_factor: float = Field(
        DEFAULT_UNIT_MARGIN_FACTOR,
        ge=0,
        description="Multiplicador del Unit_Cost para estimar ingresos retail.",
    )
    include_details: bool = Field(
        False,
        description="Si es verdadero, se regresa un desglose por vuelo/producto.",
    )
    max_details: Optional[int] = Field(
        200,
        ge=1,
        description="Numero maximo de filas de detalle a devolver cuando include_details es verdadero.",
    )


class FinancialImpactDetail(BaseModel):
    flight_id: str
    product_id: str
    product_name: Optional[str]
    service_type: Optional[str]
    unit_cost: float
    quantity_consumed: float
    standard_spec_qty: float
    recommended_load: float
    baseline_returns: float
    spir_returns: float
    waste_cost_current: float
    waste_cost_spir: float
    fuel_weight_saved_kg: float
    lost_units_recovered: float


class FinancialImpactResponse(BaseModel):
    assumptions: FinancialImpactRequest
    waste_cost_baseline: float
    waste_cost_spir: float
    waste_savings: float
    fuel_weight_reduction_kg: float
    fuel_cost_savings: float
    recovered_retail_value: float
    total_impact: float
    details: Optional[List[FinancialImpactDetail]] = Field(
        None, description="Detalle agregado por vuelo y producto."
    )
