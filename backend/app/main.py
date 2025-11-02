from __future__ import annotations

import io
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    DemandRequest,
    DemandResponse,
    GeminiSimulationRequest,
    GeminiSimulationResponse,
    SimulationPlantResult,
    StaffingRequest,
    StaffingResponse,
    FinancialImpactRequest,
    FinancialImpactResponse,
    FinancialImpactDetail,
    FinancialImpactDelta,
    WhatIfScenarioRequest,
    WhatIfScenarioResponse,
    SpeechRequest,
    SoundEffectRequest,
    FlightListResponse,
    FlightSummary,
    LotRecommendation,
    LotRecommendationResponse,
)
# Robust import of the sibling gemini module for both run modes
# (backend.app.main) and (app.main)
try:
    from .. import gemini  # when running as backend.app.main
except Exception:
    try:
        from backend import gemini  # when invoked from repo root
    except Exception:
        import gemini  # when cwd is backend/
from .financial import calculate_financial_impact, build_scenario_frames
from . import audio
from . import financial as financial_mod

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = MODELS_DIR / "global_flights_forecasting_model.pkl"

# Intentar cargar artefactos sin romper el arranque
_artifacts = {}
try:
    if ARTIFACT_PATH.exists():
        _artifacts = joblib.load(ARTIFACT_PATH)
except Exception:
    _artifacts = {}

FLIGHT_MODEL = _artifacts.get("flights_model")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://spir.tech",
        "https://www.spir.tech",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    if FLIGHT_MODEL is None:
        baseline_guess = (
            req.historical_avg_flights if req.historical_avg_flights is not None else DEFAULT_FLIGHT_BASELINE
        )
        predicted = float(baseline_guess or 0.0)
    else:
        predicted = float(FLIGHT_MODEL.predict(feature_frame)[0])
        predicted = max(predicted, 0.0)
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
        recommended_staff = max(recommended_staff, 0)

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
    predicted = max(predicted, 0.0)
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
        recommended_quantity = max(recommended_quantity, 0)

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

def _build_financial_response(
    results,
    request: FinancialImpactRequest,
    include_details_override: Optional[bool] = None,
) -> FinancialImpactResponse:
    include_flag = (
        request.include_details if include_details_override is None else include_details_override
    )
    effective_request = request.copy(update={"include_details": include_flag})

    waste_multiplier = effective_request.waste_cost_multiplier
    waste_baseline = results.waste_cost_baseline * waste_multiplier
    waste_spir = results.waste_cost_spir * waste_multiplier
    waste_savings = results.waste_savings * waste_multiplier

    details_payload = None
    if include_flag and not results.details.empty:
        max_rows = effective_request.max_details or 200
        frame = results.details.head(max_rows).copy()
        frame["waste_cost_current"] = frame["waste_cost_current"] * waste_multiplier
        frame["waste_cost_spir"] = frame["waste_cost_spir"] * waste_multiplier
        detail_rows: list[FinancialImpactDetail] = []
        for row in frame.itertuples(index=False):
            detail_rows.append(
                FinancialImpactDetail(
                    flight_id=str(getattr(row, "Flight_ID", "")),
                    product_id=str(getattr(row, "Product_ID", "")),
                    product_name=getattr(row, "Product_Name", None),
                    service_type=getattr(row, "Service_Type", None),
                    unit_cost=float(getattr(row, "Unit_Cost", 0.0) or 0.0),
                    quantity_consumed=float(getattr(row, "Quantity_Consumed", 0.0) or 0.0),
                    standard_spec_qty=float(getattr(row, "Standard_Specification_Qty", 0.0) or 0.0),
                    recommended_load=float(getattr(row, "recommended_load", 0.0) or 0.0),
                    baseline_returns=float(getattr(row, "baseline_returns", 0.0) or 0.0),
                    spir_returns=float(getattr(row, "spir_returns", 0.0) or 0.0),
                    waste_cost_current=float(getattr(row, "waste_cost_current", 0.0) or 0.0),
                    waste_cost_spir=float(getattr(row, "waste_cost_spir", 0.0) or 0.0),
                    fuel_weight_saved_kg=float(getattr(row, "fuel_weight_saved_kg", 0.0) or 0.0),
                    lost_units_recovered=float(getattr(row, "lost_units_recovered", 0.0) or 0.0),
                )
            )
        details_payload = detail_rows

    return FinancialImpactResponse(
        assumptions=effective_request,
        waste_cost_baseline=waste_baseline,
        waste_cost_spir=waste_spir,
        waste_savings=waste_savings,
        fuel_weight_reduction_kg=results.fuel_weight_reduction_kg,
        fuel_cost_savings=results.fuel_cost_savings,
        recovered_retail_value=results.recovered_retail_value,
        total_impact=waste_savings
        + results.fuel_cost_savings
        + results.recovered_retail_value,
        details=details_payload,
    )


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
    "/predict/financial-impact",
    response_model=FinancialImpactResponse,
    tags=["financial"],
)
def predict_financial_impact(request: FinancialImpactRequest) -> FinancialImpactResponse:
    try:
        dataset_override = None
        if request.origin or request.flight_id or request.product_ids:
            try:
                df = financial_mod.load_consumption_dataset()
                if request.origin:
                    df = df[df["Origin"].astype(str) == str(request.origin)]
                if request.flight_id:
                    df = df[df["Flight_ID"].astype(str) == str(request.flight_id)]
                if request.product_ids:
                    df = df[df["Product_ID"].astype(str).isin([str(x) for x in request.product_ids])]
                dataset_override = df
            except FileNotFoundError:
                # Si no hay dataset disponible, continuar sin override (se devolverán ceros)
                dataset_override = None

        results = calculate_financial_impact(
            fuel_cost_per_liter=request.fuel_cost_per_liter,
            fuel_burn_liters_per_kg=request.fuel_burn_liters_per_kg,
            buffer_factor=request.buffer_factor,
            unit_margin_factor=request.unit_margin_factor,
            dataset_override=dataset_override,
        )
    except FileNotFoundError:
        # Salvaguarda final: respuesta vacía válida
        from .financial import FinancialResults
        results = FinancialResults(
            waste_cost_baseline=0.0,
            waste_cost_spir=0.0,
            waste_savings=0.0,
            fuel_weight_reduction_kg=0.0,
            fuel_cost_savings=0.0,
            recovered_retail_value=0.0,
            details=pd.DataFrame(),
        )

    return _build_financial_response(results, request)

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
            historical_avg_flights=plant_payload.historical_avg_flights,
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
                historical_avg_passengers=plant_payload.historical_avg_passengers,
            )
            try:
                demand_prediction = _predict_passengers(demand_req)
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise

        # Ajustar predicciones con datos generados por Gemini cuando haya overrides
        gemini_total_passengers = gemini_payload.get("passengers")
        if (
            demand_prediction
            and gemini_total_passengers is not None
            and plant_payload.historical_avg_passengers
        ):
            baseline = plant_payload.historical_avg_passengers
            demand_prediction.predicted_passengers = float(gemini_total_passengers)
            demand_prediction.baseline_passengers = float(baseline)
            demand_prediction.demand_index = (
                float(gemini_total_passengers) / baseline if baseline > 0 else None
            )
            if (
                demand_prediction.demand_index is not None
                and plant_payload.base_quantity is not None
            ):
                demand_prediction.recommended_quantity = max(
                    math.ceil(
                        plant_payload.base_quantity * demand_prediction.demand_index
                    ),
                    0,
                )

        gemini_total_flights = gemini_payload.get("flights")
        if (
            staffing_prediction
            and gemini_total_flights is not None
            and plant_payload.historical_avg_flights
        ):
            baseline = plant_payload.historical_avg_flights
            staffing_prediction.predicted_flights = float(gemini_total_flights)
            staffing_prediction.baseline_flights = float(baseline)
            staffing_prediction.workload_index = (
                float(gemini_total_flights) / baseline if baseline > 0 else None
            )
            if (
                staffing_prediction.workload_index is not None
                and plant_payload.staff_baseline is not None
            ):
                staffing_prediction.recommended_staff = max(
                    math.ceil(
                        plant_payload.staff_baseline
                        * staffing_prediction.workload_index
                    ),
                    0,
                )

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
@app.post(
    "/simulate/what-if",
    response_model=WhatIfScenarioResponse,
    tags=["simulation"],
    summary="Genera un escenario con Gemini y cuantifica su impacto financiero.",
)
def simulate_what_if(request: WhatIfScenarioRequest) -> WhatIfScenarioResponse:
    if not request.plants:
        raise HTTPException(status_code=400, detail="Se requiere al menos una planta para la simulacion.")

    scenario_date = request.date
    assumptions = request.financial_assumptions or FinancialImpactRequest()
    plant_results: List[SimulationPlantResult] = []
    normalized_payloads: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for plant_payload in request.plants:
        target_date = plant_payload.target_date or scenario_date
        target_date_str = target_date.strftime("%Y-%m-%d")

        prompt_hints = {
            "name": plant_payload.name,
            "baseline_passengers": plant_payload.historical_avg_passengers,
            "baseline_flights": plant_payload.historical_avg_flights,
            "notes": plant_payload.description,
        }
        prompt = gemini.build_scenario_prompt(
            plant_payload.plant_id,
            request.scenario,
            target_date_str,
            hints={k: v for k, v in prompt_hints.items() if v},
        )

        gemini_raw = None
        if request.use_gemini:
            gemini_raw = gemini.generate_flight_data(
                plant_payload.plant_id,
                target_date_str,
                prompt=prompt,
                skip_db=True,
            )
        if not gemini_raw:
            fallback_row = gemini.generate_example_row(plant_payload.plant_id, target_date_str)
            if fallback_row:
                fallback_row.setdefault("day", target_date_str.replace("-", ""))
                gemini_raw = fallback_row
                warnings.append(
                    f"Se utilizo una fila sintetica de respaldo para la planta {plant_payload.plant_id}."
                )
            else:
                warnings.append(
                    f"No se pudieron generar datos con Gemini para la planta {plant_payload.plant_id}."
                )
                continue

        gemini_payload = gemini.prepare_api_payload(gemini_raw, plant_payload.plant_id)
        gemini_payload["scenario_text"] = request.scenario
        normalized_payloads.append(
            {
                **gemini_payload,
                "baseline_passengers": plant_payload.historical_avg_passengers,
                "baseline_flights": plant_payload.historical_avg_flights,
            }
        )

        prediction_date = _parse_day_str(gemini_payload.get("day")) or target_date

        staffing_req = StaffingRequest(
            date=prediction_date,
            plant_id=plant_payload.plant_id,
            staff_baseline=plant_payload.staff_baseline,
            historical_avg_flights=plant_payload.historical_avg_flights,
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
                historical_avg_passengers=plant_payload.historical_avg_passengers,
            )
            try:
                demand_prediction = _predict_passengers(demand_req)
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise

        gemini_total_passengers = gemini_payload.get("passengers")
        if (
            demand_prediction
            and gemini_total_passengers is not None
            and plant_payload.historical_avg_passengers
        ):
            baseline = plant_payload.historical_avg_passengers
            demand_prediction.predicted_passengers = float(gemini_total_passengers)
            demand_prediction.baseline_passengers = float(baseline)
            demand_prediction.demand_index = (
                float(gemini_total_passengers) / baseline if baseline > 0 else None
            )
            if (
                demand_prediction.demand_index is not None
                and plant_payload.base_quantity is not None
            ):
                demand_prediction.recommended_quantity = max(
                    math.ceil(
                        plant_payload.base_quantity * demand_prediction.demand_index
                    ),
                    0,
                )

        gemini_total_flights = gemini_payload.get("flights")
        if (
            staffing_prediction
            and gemini_total_flights is not None
            and plant_payload.historical_avg_flights
        ):
            baseline = plant_payload.historical_avg_flights
            staffing_prediction.predicted_flights = float(gemini_total_flights)
            staffing_prediction.baseline_flights = float(baseline)
            staffing_prediction.workload_index = (
                float(gemini_total_flights) / baseline if baseline > 0 else None
            )
            if (
                staffing_prediction.workload_index is not None
                and plant_payload.staff_baseline is not None
            ):
                staffing_prediction.recommended_staff = max(
                    math.ceil(
                        plant_payload.staff_baseline
                        * staffing_prediction.workload_index
                    ),
                    0,
                )

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
        plant_results.append(plant_result)

    if not plant_results:
        raise HTTPException(
            status_code=502,
            detail="No se genero ninguna simulacion valida con Gemini.",
        )

    baseline_df, scenario_df, frame_warnings = build_scenario_frames(normalized_payloads)
    warnings.extend(frame_warnings)

    baseline_results = calculate_financial_impact(
        fuel_cost_per_liter=assumptions.fuel_cost_per_liter,
        fuel_burn_liters_per_kg=assumptions.fuel_burn_liters_per_kg,
        buffer_factor=assumptions.buffer_factor,
        unit_margin_factor=assumptions.unit_margin_factor,
        dataset_override=baseline_df,
    )
    scenario_results = calculate_financial_impact(
        fuel_cost_per_liter=assumptions.fuel_cost_per_liter,
        fuel_burn_liters_per_kg=assumptions.fuel_burn_liters_per_kg,
        buffer_factor=assumptions.buffer_factor,
        unit_margin_factor=assumptions.unit_margin_factor,
        dataset_override=scenario_df,
    )

    baseline_response = _build_financial_response(baseline_results, assumptions, include_details_override=False)
    scenario_response = _build_financial_response(scenario_results, assumptions)

    financial_delta = FinancialImpactDelta(
        waste_cost_baseline=scenario_response.waste_cost_baseline - baseline_response.waste_cost_baseline,
        waste_cost_spir=scenario_response.waste_cost_spir - baseline_response.waste_cost_spir,
        waste_savings=scenario_response.waste_savings - baseline_response.waste_savings,
        fuel_cost_savings=scenario_response.fuel_cost_savings - baseline_response.fuel_cost_savings,
        recovered_retail_value=scenario_response.recovered_retail_value - baseline_response.recovered_retail_value,
        total_impact=scenario_response.total_impact - baseline_response.total_impact,
    )

    return WhatIfScenarioResponse(
        scenario=request.scenario,
        date=request.date,
        plant_results=plant_results,
        financial_baseline=baseline_response,
        financial_scenario=scenario_response,
        financial_delta=financial_delta,
        warnings=warnings,
        gemini_metadata={"scenario_text": request.scenario, **request.gemini_metadata},
    )


@app.get(
    "/flights/list",
    response_model=FlightListResponse,
    tags=["flights"],
    summary="Devuelve una lista de vuelos recientes basada en el dataset de consumo.",
)
def list_flights() -> FlightListResponse:
    df = financial_mod.load_consumption_dataset()
    # Tomar la fecha mas reciente
    df["Date"] = pd.to_datetime(df["Date"])  # type: ignore[index]
    latest_date = df["Date"].max()
    sample = (
        df[df["Date"] == latest_date]
        .groupby(["Flight_ID", "Origin"], as_index=False)["Passenger_Count"]
        .sum()
        .head(12)
    )

    flights: list[FlightSummary] = []
    for row in sample.itertuples(index=False):
        flight_id = str(getattr(row, "Flight_ID"))
        origin = str(getattr(row, "Origin"))
        passengers = int(getattr(row, "Passenger_Count") or 0)
        airline = "gategroup partner"
        route = f"{origin} - HUB"
        flights.append(
            FlightSummary(
                flight_id=flight_id,
                airline=airline,
                route=route,
                date=str(latest_date.date()),
                departure_time="--:--",
                plant_id=1,
                origin=origin,
                destination="HUB",
                passengers=passengers,
                flights=None,
            )
        )

    return FlightListResponse(flights=flights)


@app.get(
    "/lots/recommend",
    response_model=LotRecommendationResponse,
    tags=["lots"],
    summary="Sugiere lotes y productos para un vuelo.",
)
def recommend_lots(flight_id: str, origin: str) -> LotRecommendationResponse:
    df = financial_mod.load_consumption_dataset()
    df_exp = financial_mod.load_expiration_dataset()

    df_flight = df[(df["Flight_ID"].astype(str) == str(flight_id)) & (df["Origin"].astype(str) == str(origin))]
    if df_flight.empty:
        # fallback: usar por origen solamente
        df_flight = df[df["Origin"].astype(str) == str(origin)].copy()

    # Unir expiraciones por Product_ID
    merged = df_flight.merge(
        df_exp[["Product_ID", "Product_Name", "LOT_Number", "Expiry_Date"]],
        on="Product_ID",
        how="left",
    )

    # Priorizar menor caducidad
    merged["Expiry_Date"] = pd.to_datetime(merged["Expiry_Date"], errors="coerce")
    merged = merged.sort_values(["Expiry_Date", "Unit_Cost"])  # type: ignore[index]
    merged = merged.head(12)

    lots: list[LotRecommendation] = []
    def _get(row, names):
        for n in names:
            if hasattr(row, n):
                v = getattr(row, n)
                if v is not None:
                    return v
        return None

    for i, row in enumerate(merged.itertuples(index=False)):
        product_name = _get(row, ["Product_Name", "Product_Name_x", "Product_Name_y"]) or ""
        lot_number = _get(row, ["LOT_Number", "LOT_Number_x", "LOT_Number_y"]) 
        expiry_val = _get(row, ["Expiry_Date", "Expiry_Date_x", "Expiry_Date_y"]) 
        lots.append(
            LotRecommendation(
                product_id=str(getattr(row, "Product_ID")),
                product_name=str(product_name),
                lot_number=(str(lot_number) if lot_number is not None else None),
                expiry_date=(str(expiry_val) if expiry_val is not None else None),
                standard_spec_qty=float(getattr(row, "Standard_Specification_Qty") or 0.0),
                quantity_consumed=float(getattr(row, "Quantity_Consumed") or 0.0),
                unit_cost=float(getattr(row, "Unit_Cost") or 0.0),
                service_type=(str(getattr(row, "Service_Type")) if getattr(row, "Service_Type", None) is not None else None),
                crew_feedback=(
                    str(getattr(row, "Crew_Feedback")) if getattr(row, "Crew_Feedback", None) is not None else None
                ),
                recommended=(i == 0),
            )
        )

    return LotRecommendationResponse(flight_id=flight_id, origin=origin, lots=lots)


def _voice_settings_from_request(request: SpeechRequest) -> Optional[Dict[str, float]]:
    if request.stability is None and request.similarity_boost is None:
        return None
    return {
        "stability": request.stability if request.stability is not None else 0.3,
        "similarity_boost": request.similarity_boost if request.similarity_boost is not None else 0.7,
    }


@app.post(
    "/assist/speak",
    tags=["assistive"],
    summary="Genera instrucciones por voz usando ElevenLabs.",
)
async def assistive_speak(request: SpeechRequest) -> StreamingResponse:
    audio_bytes = await audio.synthesize_speech(
        request.text,
        voice_id=request.voice_id,
        model_id=request.model_id,
        voice_settings=_voice_settings_from_request(request),
    )

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post(
    "/assist/sound",
    tags=["assistive"],
    summary="Genera efectos de sonido breves para feedback auditivo.",
)
async def assistive_sound(request: SoundEffectRequest) -> StreamingResponse:
    audio_bytes = await audio.generate_sound_effect(
        request.prompt,
        duration_seconds=request.duration_seconds,
        model_id=request.model_id,
    )

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/flights", response_model=FlightListResponse, tags=["demo"])
def list_flights_demo() -> FlightListResponse:
    """Endpoint demo que devuelve una lista de vuelos sinteticos para la UI."""
    sample = [
        {
            "flight_id": "AM109",
            "airline": "Aeromexico",
            "route": "MEX -> DOH",
            "date": "2025-10-27",
            "departure_time": "08:45",
            "plant_id": 1,
            "origin": "MEX",
            "destination": "DOH",
            "passengers": 210,
            "flights": 1,
            "base_quantity": 120.0,
            "staff_baseline": 42,
        },
        {
            "flight_id": "LH432",
            "airline": "Lufthansa",
            "route": "FRA -> JFK",
            "date": "2025-10-27",
            "departure_time": "13:20",
            "plant_id": 3,
            "origin": "FRA",
            "destination": "JFK",
            "passengers": 320,
            "flights": 1,
            "base_quantity": 200.0,
            "staff_baseline": 55,
        },
        {
            "flight_id": "BA249",
            "airline": "British Airways",
            "route": "LHR -> GRU",
            "date": "2025-10-27",
            "departure_time": "18:10",
            "plant_id": 4,
            "origin": "LHR",
            "destination": "GRU",
            "passengers": 280,
            "flights": 1,
            "base_quantity": 180.0,
            "staff_baseline": 48,
        },
        {
            "flight_id": "EK214",
            "airline": "Emirates",
            "route": "IAH -> DXB",
            "date": "2025-10-28",
            "departure_time": "21:55",
            "plant_id": 2,
            "origin": "IAH",
            "destination": "DXB",
            "passengers": 360,
            "flights": 1,
            "base_quantity": 220.0,
            "staff_baseline": 60,
        },
    ]

    return FlightListResponse(flights=sample)


@app.get("/lots/recommend/demo", response_model=LotRecommendationResponse, tags=["demo"])
def recommend_lots_demo(flight_id: str) -> LotRecommendationResponse:
    """Endpoint demo que devuelve lotes recomendados para un vuelo dado."""
    # In a real implementation we'd query inventory and scoring logic. Here we return examples.
    lots = [
        {
            "product_id": "P-CHIPS-50",
            "product_name": "Chips Clásicas 50g",
            "lot_number": "B47-2025",
            "expiry_date": "2025-11-12",
            "standard_spec_qty": 12.0,
            "quantity_consumed": 8.0,
            "unit_cost": 6.5,
            "service_type": "Retail",
            "crew_feedback": "Buena aceptación",
            "recommended": True,
        },
        {
            "product_id": "P-EBAR-60",
            "product_name": "Energy Bar 60g",
            "lot_number": "C12-2025",
            "expiry_date": "2025-11-18",
            "standard_spec_qty": 10.0,
            "quantity_consumed": 6.0,
            "unit_cost": 12.0,
            "service_type": "Crew",
            "crew_feedback": "Preferido por tripulación",
            "recommended": False,
        },
    ]

    origin = "MEX"
    return LotRecommendationResponse(flight_id=flight_id, origin=origin, lots=lots)



