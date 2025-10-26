"""Quantifies the financial impact of the consumption model for the judges."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

import joblib
import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "external"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "consumption_prediction_xgb.pkl"

PLANT_ORIGIN_MAP = {
    1: "DOH",
    2: "JFK",
    3: "LHR",
    4: "MEX",
    5: "NRT",
    6: "ZRH",
}

FUEL_BURN_LITERS_PER_KG = 0.0003  # liters saved per kilogram removed
DEFAULT_FUEL_COST_PER_LITER = 25.0
DEFAULT_UNIT_MARGIN_FACTOR = 1.0  # multiplier over Unit_Cost to estimate retail value
RECOMMENDED_BUFFER = 1.05  # safety buffer applied to predicted consumption

def _parse_weight(value: object) -> float:
    """Return weight in grams extracted from strings like '150g', '0.2kg', '250ml'."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return np.nan
    weight = float(match.group(1))
    if "kg" in text:
        weight *= 1000.0
    elif "l" in text and "ml" not in text:
        weight *= 1000.0
    return weight


def _load_consumption_dataset() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "consumption_prediction.xlsx",
        PROJECT_ROOT / "data" / "consumption_prediction.xlsx",
        PROJECT_ROOT
        / "Gategroup 2025-20251025T180038Z-1-001"
        / "Gategroup 2025"
        / "HackMTY2025_ChallengeDimensions"
        / "02_ConsumptionPrediction"
        / "[HackMTY2025]_ConsumptionPrediction_Dataset_v1.xlsx",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_excel(path)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
    raise FileNotFoundError("Missing consumption dataset in data/external/consumption_prediction.xlsx.")


def _expiration_candidates() -> List[Path]:
    return [
        DATA_DIR / "expiration_management.xlsx",
        PROJECT_ROOT / "data" / "expiration_management.xlsx",
        PROJECT_ROOT
        / "Gategroup 2025-20251025T180038Z-1-001"
        / "Gategroup 2025"
        / "HackMTY2025_ChallengeDimensions"
        / "01_ExpirationDateManagement"
        / "[HackMTY2025]_ExpirationDateManagement_Dataset_v1.xlsx",
    ]


def load_expiration_dataset() -> pd.DataFrame:
    for path in _expiration_candidates():
        if path.exists():
            return pd.read_excel(path)
    raise FileNotFoundError("Missing expiration dataset in data/external/expiration_management.xlsx.")


def _load_product_catalog() -> pd.DataFrame:
    df = load_expiration_dataset()
    df["Weight_or_Volume"] = df["Weight_or_Volume"].apply(_parse_weight)
    catalog = (
        df.groupby(["Product_ID", "Product_Name"])["Weight_or_Volume"]
        .mean()
        .reset_index()
        .rename(columns={"Weight_or_Volume": "product_weight"})
    )
    return catalog


def _load_flight_complexity() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "productivity_estimation.xlsx",
        PROJECT_ROOT / "data" / "productivity_estimation.xlsx",
        PROJECT_ROOT
        / "Gategroup 2025-20251025T180038Z-1-001"
        / "Gategroup 2025"
        / "HackMTY2025_ChallengeDimensions"
        / "03_ProductivityEstimation"
        / "[HackMTY2025]_ProductivityEstimation_Dataset_v1.xlsx",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_excel(path)
            for column in ["Total_Items", "Unique_Item_Types"]:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            complexity = (
                df.groupby("Flight_Type", as_index=False)
                .agg({"Total_Items": "mean", "Unique_Item_Types": "mean"})
                .rename(
                    columns={
                        "Total_Items": "flight_avg_total_items",
                        "Unique_Item_Types": "flight_avg_unique_items",
                    }
                )
            )
            return complexity
    raise FileNotFoundError("Missing productivity dataset in data/external/productivity_estimation.xlsx.")

def build_scenario_frames(payloads: Iterable[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_df = _load_consumption_dataset()
    baseline_frames: List[pd.DataFrame] = []
    scenario_frames: List[pd.DataFrame] = []
    warnings: List[str] = []

    for payload in payloads:
        plant_id = int(payload.get("plant_id", 0) or 0)
        origin = PLANT_ORIGIN_MAP.get(plant_id)
        if not origin:
            warnings.append(f"Planta {plant_id} no tiene origen mapeado.")
            continue

        plant_rows = base_df[base_df["Origin"] == origin]
        if plant_rows.empty:
            warnings.append(f"No hay filas base para la planta {plant_id} (origen {origin}).")
            continue

        baseline_frames.append(plant_rows)

        scenario_copy = plant_rows.copy()
        day_value = str(payload.get("day", "")).strip()
        scenario_date = pd.to_datetime(day_value, format="%Y%m%d", errors="coerce")
        if pd.isna(scenario_date):
            scenario_date = pd.Timestamp.utcnow().normalize()
        scenario_copy["Date"] = scenario_date

        passengers = payload.get("passengers")
        if passengers is None or passengers <= 0:
            passengers = float(plant_rows["Passenger_Count"].mean())
        baseline_passengers = payload.get("baseline_passengers") or float(
            plant_rows["Passenger_Count"].mean() or 0.0
        )
        ratio = (
            float(passengers) / baseline_passengers if baseline_passengers and baseline_passengers > 0 else 1.0
        )

        scenario_copy["Passenger_Count"] = float(passengers)
        scenario_copy["Flight_ID"] = scenario_copy["Flight_ID"].astype(str) + "_SIM"

        for col in ["Standard_Specification_Qty", "Quantity_Returned", "Quantity_Consumed"]:
            scenario_copy[col] = (
                scenario_copy[col].astype(float) * ratio
            ).round().clip(lower=0).astype(int)

        scenario_copy["Crew_Feedback"] = None
        scenario_frames.append(scenario_copy)

    baseline_df = (
        pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    )
    scenario_df = (
        pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    )
    return baseline_df, scenario_df, warnings

@dataclass
class FeatureBundle:
    features: Iterable[str]
    categorical: Iterable[str]
    numeric: Iterable[str]


def _build_feature_frame(
    df: pd.DataFrame,
    product_catalog: pd.DataFrame,
    flight_complexity: pd.DataFrame,
    bundle: FeatureBundle,
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_df = df.copy()

    numeric_columns = [
        "Passenger_Count",
        "Standard_Specification_Qty",
        "Quantity_Returned",
        "Quantity_Consumed",
        "Unit_Cost",
    ]
    for column in numeric_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

    feature_df["dayofweek"] = feature_df["Date"].dt.dayofweek
    feature_df["month"] = feature_df["Date"].dt.month
    feature_df["weekofyear"] = feature_df["Date"].dt.isocalendar().week.astype(int)
    feature_df["dayofyear"] = feature_df["Date"].dt.dayofyear
    feature_df["is_weekend"] = (feature_df["dayofweek"] >= 5).astype(int)

    passenger_for_ratios = feature_df["Passenger_Count"].replace(0, np.nan)
    spec_for_ratios = feature_df["Standard_Specification_Qty"].replace(0, np.nan)

    feature_df["spec_per_passenger"] = (
        spec_for_ratios / passenger_for_ratios
    ).replace([np.inf, -np.inf], np.nan)
    feature_df["spec_per_passenger"] = feature_df["spec_per_passenger"].fillna(
        feature_df["spec_per_passenger"].median()
    )

    weight_median = product_catalog["product_weight"].median()
    feature_df = feature_df.merge(
        product_catalog[["Product_ID", "product_weight"]], on="Product_ID", how="left"
    )
    feature_df["product_weight"] = feature_df["product_weight"].fillna(weight_median)

    feature_df = feature_df.merge(flight_complexity, on="Flight_Type", how="left")
    for column in ["flight_avg_total_items", "flight_avg_unique_items"]:
        feature_df[column] = feature_df[column].fillna(feature_df[column].mean())

    feature_df["consumption_per_passenger"] = (
        feature_df["Quantity_Consumed"] / passenger_for_ratios
    ).replace([np.inf, -np.inf], np.nan)
    feature_df["return_ratio"] = (
        feature_df["Quantity_Returned"] / spec_for_ratios
    ).replace([np.inf, -np.inf], np.nan)
    feature_df["consumed_ratio"] = (
        feature_df["Quantity_Consumed"] / spec_for_ratios
    ).replace([np.inf, -np.inf], np.nan)

    feature_df["weight_per_passenger"] = (
        feature_df["product_weight"] * feature_df["consumption_per_passenger"]
    )
    feature_df["weight_per_spec"] = (
        feature_df["product_weight"] * feature_df["Standard_Specification_Qty"]
    )

    for col in [
        "consumption_per_passenger",
        "return_ratio",
        "consumed_ratio",
        "weight_per_passenger",
    ]:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    target = feature_df["Quantity_Consumed"]
    X = feature_df[list(bundle.features)]
    return X, target


def _predict_with_service_models(
    X: pd.DataFrame,
    df_meta: pd.DataFrame,
    bundle: FeatureBundle,
    global_model,
    service_models: Dict[str, Dict[str, object]],
) -> np.ndarray:
    preds = np.zeros(len(X), dtype=float)
    for service_type in df_meta["Service_Type"].unique():
        mask = df_meta["Service_Type"] == service_type
        if mask.sum() == 0:
            continue
        info = service_models.get(service_type)
        model = info["model"] if info else global_model
        mask_np = mask.to_numpy()
        preds[mask_np] = model.predict(X.iloc[mask_np, :][list(bundle.features)])
    return preds


@dataclass
class FinancialResults:
    waste_cost_baseline: float
    waste_cost_spir: float
    waste_savings: float
    fuel_weight_reduction_kg: float
    fuel_cost_savings: float
    recovered_retail_value: float
    details: pd.DataFrame

    @property
    def total_impact(self) -> float:
        return (
            self.waste_savings + self.fuel_cost_savings + self.recovered_retail_value
        )


def calculate_financial_impact(
    fuel_cost_per_liter: float = DEFAULT_FUEL_COST_PER_LITER,
    fuel_burn_liters_per_kg: float = FUEL_BURN_LITERS_PER_KG,
    buffer_factor: float = RECOMMENDED_BUFFER,
    unit_margin_factor: float = DEFAULT_UNIT_MARGIN_FACTOR,
    dataset_override: Optional[pd.DataFrame] = None,
) -> FinancialResults:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Train the consumption model first."
        )

    artifact = joblib.load(MODEL_PATH)
    bundle = FeatureBundle(
        features=artifact["features"],
        categorical=artifact["categorical_features"],
        numeric=artifact["numeric_features"],
    )
    global_model = artifact["model"]
    service_models = artifact.get("service_models", {})

    if dataset_override is not None:
        if dataset_override.empty:
            return FinancialResults(
                waste_cost_baseline=0.0,
                waste_cost_spir=0.0,
                waste_savings=0.0,
                fuel_weight_reduction_kg=0.0,
                fuel_cost_savings=0.0,
                recovered_retail_value=0.0,
                details=pd.DataFrame(),
            )
        df = dataset_override.copy()
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df = _load_consumption_dataset()

    product_catalog = _load_product_catalog()
    flight_complexity = _load_flight_complexity()

    df_merged = df.merge(
        product_catalog[["Product_ID", "product_weight"]], on="Product_ID", how="left"
    )

    X, _ = _build_feature_frame(df, product_catalog, flight_complexity, bundle)
    predictions = _predict_with_service_models(
        X, df, bundle, global_model, service_models
    )

    df_merged["predicted_consumption"] = predictions
    df_merged["recommended_load"] = np.maximum(
        df_merged["Quantity_Consumed"], predictions * buffer_factor
    )

    baseline_returns = df_merged["Quantity_Returned"].clip(lower=0)
    baseline_waste_cost = float((baseline_returns * df_merged["Unit_Cost"]).sum())

    spir_returns = (
        df_merged["recommended_load"] - df_merged["Quantity_Consumed"]
    ).clip(lower=0)
    spir_waste_cost = float((spir_returns * df_merged["Unit_Cost"]).sum())

    product_weight_kg = df_merged["product_weight"].fillna(
        product_catalog["product_weight"].median()
    ) / 1000.0

    weight_difference = (
        df_merged["Standard_Specification_Qty"] - df_merged["recommended_load"]
    ).clip(lower=0)
    fuel_weight_saved_kg = float((weight_difference * product_weight_kg).sum())
    fuel_liters_saved = fuel_weight_saved_kg * fuel_burn_liters_per_kg
    fuel_cost_savings = fuel_liters_saved * fuel_cost_per_liter

    ran_out_mask = (
        df_merged["Crew_Feedback"].astype(str).str.lower().str.contains("ran out")
    )
    potential_demand = df_merged.loc[ran_out_mask, "recommended_load"]
    current_spec = df_merged.loc[ran_out_mask, "Standard_Specification_Qty"]
    lost_units = (potential_demand - current_spec).clip(lower=0)
    recovered_retail_value = float(
        (lost_units * df_merged.loc[ran_out_mask, "Unit_Cost"] * unit_margin_factor).sum()
    )

    details = df_merged.assign(
        baseline_returns=baseline_returns,
        spir_returns=spir_returns,
        weight_difference_units=weight_difference,
        fuel_weight_saved_kg=weight_difference * product_weight_kg,
        lost_units_recovered=0.0,
    )
    if ran_out_mask.any():
        details.loc[ran_out_mask, "lost_units_recovered"] = lost_units
    details["waste_cost_current"] = details["baseline_returns"] * details["Unit_Cost"]
    details["waste_cost_spir"] = details["spir_returns"] * details["Unit_Cost"]

    return FinancialResults(
        waste_cost_baseline=baseline_waste_cost,
        waste_cost_spir=spir_waste_cost,
        waste_savings=baseline_waste_cost - spir_waste_cost,
        fuel_weight_reduction_kg=fuel_weight_saved_kg,
        fuel_cost_savings=fuel_cost_savings,
        recovered_retail_value=recovered_retail_value,
        details=details,
    )

def resumen_financiero() -> None:
    results = calculate_financial_impact()
    print("=== Impacto financiero estimado (análisis piloto) ===")
    print(f"Ahorro por desperdicio:        ${results.waste_savings:,.2f}")
    print(
        f"Ahorro coste combustible:      ${results.fuel_cost_savings:,.2f} "
        f"(peso evitado: {results.fuel_weight_reduction_kg:,.1f} kg)"
    )
    print(f"Ingresos recuperados retail:   ${results.recovered_retail_value:,.2f}")
    print(f"Impacto total estimado:        ${results.total_impact:,.2f}")


def load_consumption_dataset() -> pd.DataFrame:
    return _load_consumption_dataset()


__all__ = [
    "calculate_financial_impact",
    "FinancialResults",
    "FUEL_BURN_LITERS_PER_KG",
    "DEFAULT_FUEL_COST_PER_LITER",
    "DEFAULT_UNIT_MARGIN_FACTOR",
    "RECOMMENDED_BUFFER",
    "resumen_financiero",
    "load_consumption_dataset",
    "load_expiration_dataset",
]
