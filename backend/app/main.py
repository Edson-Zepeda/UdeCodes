from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from .models import (
    DemandRequest,
    DemandResponse,
    GeminiSimulationRequest,
    GeminiSimulationResponse,
    SimulationPlantResult,
    StaffingRequest,
    StaffingResponse,
)
from .. import gemini

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = MODELS_DIR / "global_flights_forecasting_model.pkl"

if not ARTIFACT_PATH.exists():
    raise RuntimeError(
        f"No se encontro el archivo de artefactos del modelo en {ARTIFACT_PATH}. "
        "Ejecuta el notebook de entrenamiento antes de iniciar el backend."
    )

_artifacts = joblib.load(ARTIFACT_PATH)

FLIGHT_MODEL = _artifacts["flights_model"]
FLIGHT_FEATURES = _artifacts.get("flight_features") or [
    "plant_id",
    "dayofweek",
    "quarter",
    "month",
    "year",
    "dayofyear",
    "weekofyear",
]
BASELINE_FLIGHTS: Dict[int, float] = {
    int(k): float(v) for k, v in _artifacts.get("flight_baseline", {}).items()
}

PASSENGER_MODEL = _artifacts.get("passenger_model")
PASSENGER_FEATURES = _artifacts.get("passenger_features") or FLIGHT_FEATURES
BASELINE_PASSENGERS: Dict[int, float] = {
    int(k): float(v) for k, v in _artifacts.get("passenger_baseline", {}).items()
}

DEFAULT_FLIGHT_BASELINE = (
    float(np.mean(list(BASELINE_FLIGHTS.values()))) if BASELINE_FLIGHTS else None
)
DEFAULT_PASSENGER_BASELINE = (
    float(np.mean(list(BASELINE_PASSENGERS.values()))) if BASELINE_PASSENGERS else None
)


def _parse_day_str(day_value: Optional[str]) -> Optional[date]:
    if not day_value:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(day_value, fmt).date()
        except ValueError:
            continue
    return None


app = FastAPI(
    title="SPIR Forecasting API",
    description="Servicios de forecasting y simulacion para el proyecto SPIR.",
    version="1.0.0",
)


def _ensure_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _compute_features(target_date: date, plant_id: int) -> pd.DataFrame:
    target_date = _ensure_date(target_date)
    iso_calendar = target_date.isocalendar()
    quarter = (target_date.month - 1) // 3 + 1

    feature_dict = {
        "plant_id": [plant_id],
        "dayofweek": [target_date.weekday()],
        "quarter": [quarter],
        "month": [target_date.month],
        "year": [target_date.year],
        "dayofyear": [target_date.timetuple().tm_yday],
        "weekofyear": [iso_calendar[1]],
    }

    df = pd.DataFrame(feature_dict)
    return df


def _predict_flights(req: StaffingRequest) -> StaffingResponse:
    feature_frame = _compute_features(req.date, req.plant_id)
    feature_frame = feature_frame[FLIGHT_FEATURES]

    predicted = float(FLIGHT_MODEL.predict(feature_frame)[0])
    baseline = _resolve_baseline(
        BASELINE_FLIGHTS,
        req.plant_id,
        req.historical_avg_flights,
        DEFAULT_FLIGHT_BASELINE,
    )

    workload_index = (predicted / baseline) if baseline and baseline > 0 else None

    recommended_staff = None
    if req.staff_baseline is not None and workload_index is not None:
        recommended_staff = math.ceil(req.staff_baseline * workload_index)

    return StaffingResponse(
        date=req.date,
        plant_id=req.plant_id,
        predicted_flights=predicted,
        baseline_flights=baseline,
        workload_index=workload_index,
        recommended_staff=recommended_staff,
    )


def _predict_passengers(req: DemandRequest) -> DemandResponse:
    if PASSENGER_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de pasajeros no esta disponible. Ejecuta el entrenamiento con datos de pasajeros.",
        )

    feature_frame = _compute_features(req.date, req.plant_id)
    feature_frame = feature_frame[PASSENGER_FEATURES]

    predicted = float(PASSENGER_MODEL.predict(feature_frame)[0])
    baseline = _resolve_baseline(
        BASELINE_PASSENGERS,
        req.plant_id,
        req.historical_avg_passengers,
        DEFAULT_PASSENGER_BASELINE,
    )

    demand_index = (predicted / baseline) if baseline and baseline > 0 else None

    recommended_quantity = None
    if req.base_quantity is not None and demand_index is not None:
        recommended_quantity = math.ceil(req.base_quantity * demand_index)

    return DemandResponse(
        date=req.date,
        plant_id=req.plant_id,
        predicted_passengers=predicted,
        baseline_passengers=baseline,
        demand_index=demand_index,
        recommended_quantity=recommended_quantity,
    )


def _resolve_baseline(
    baseline_map: Dict[int, float],
    plant_id: int,
    override_value: Optional[float],
    fallback: Optional[float],
) -> Optional[float]:
    if override_value:
        return float(override_value)
    if plant_id in baseline_map:
        return float(baseline_map[plant_id])
    return fallback


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/staffing", response_model=StaffingResponse, tags=["predictions"])
def predict_staffing(request: StaffingRequest) -> StaffingResponse:
    return _predict_flights(request)


@app.post("/predict/demand", response_model=DemandResponse, tags=["predictions"])
def predict_demand(request: DemandRequest) -> DemandResponse:
    return _predict_passengers(request)


@app.post(
    "/simulate/generative",
    response_model=GeminiSimulationResponse,
    tags=["simulation"],
    summary="Procesa un escenario generado por Gemini y devuelve predicciones SPIR.",
)
def simulate_generative(request: GeminiSimulationRequest) -> GeminiSimulationResponse:
    results: list[SimulationPlantResult] = []

    for plant_payload in request.plants:
        target_date = plant_payload.target_date or request.date
        target_date_str = target_date.strftime("%Y-%m-%d")
        prompt = gemini.build_prompt(plant_payload.plant_id, target_date_str)
        gemini_raw = gemini.generate_flight_data(
            plant_payload.plant_id,
            target_date_str,
            prompt=prompt,
        )
        if not gemini_raw:
            raise HTTPException(
                status_code=502,
                detail=f"No se pudieron generar datos con Gemini para la planta {plant_payload.plant_id}.",
            )

        gemini_payload = gemini.prepare_api_payload(gemini_raw, plant_payload.plant_id)
        prediction_date = _parse_day_str(gemini_payload.get("day")) or target_date

        staffing_req = StaffingRequest(
            date=prediction_date,
            plant_id=plant_payload.plant_id,
            staff_baseline=plant_payload.staff_baseline,
        )
        staffing_prediction = _predict_flights(staffing_req)

        demand_prediction = None
        if PASSENGER_MODEL is not None:
            base_quantity = (
                plant_payload.base_quantity
                if plant_payload.base_quantity is not None
                else gemini_payload.get("max_capacity")
            )
            demand_req = DemandRequest(
                date=prediction_date,
                plant_id=plant_payload.plant_id,
                base_quantity=base_quantity,
            )
            try:
                demand_prediction = _predict_passengers(demand_req)
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise

        plant_result = SimulationPlantResult(
            plant_id=plant_payload.plant_id,
            name=plant_payload.name,
            target_date=prediction_date,
            gemini_prompt=prompt,
            gemini_raw=gemini_raw,
            gemini_payload={
                "normalized": gemini_payload,
                "description": plant_payload.description,
                "metadata": plant_payload.metadata,
                "base_quantity": plant_payload.base_quantity,
                "staff_baseline": plant_payload.staff_baseline,
            },
            demand_prediction=demand_prediction,
            staffing_prediction=staffing_prediction,
        )
        results.append(plant_result)

    return GeminiSimulationResponse(
        date=request.date,
        scenario_id=request.scenario_id,
        results=results,
        gemini_metadata=request.gemini_metadata,
    )
