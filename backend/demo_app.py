from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Dict
import pandas as pd
import datetime
import io
import wave
import struct
import math
from fastapi.responses import Response

app = FastAPI(title="SPIR Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static demo flights (also used for origin lookup)
FLIGHTS = [
    {"flight_id": "AM109", "airline": "AeroMéxico", "route": "MEX → DOH", "date": "2025-10-27", "departure_time": "08:45", "plant_id": 1, "origin": "MEX", "destination": "DOH"},
    {"flight_id": "LH432", "airline": "Lufthansa", "route": "FRA → JFK", "date": "2025-10-27", "departure_time": "13:20", "plant_id": 3, "origin": "FRA", "destination": "JFK"},
    {"flight_id": "BA249", "airline": "British Airways", "route": "LHR → GRU", "date": "2025-10-27", "departure_time": "18:10", "plant_id": 4, "origin": "LHR", "destination": "GRU"},
]


def _expiration_paths() -> List[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "external"
    candidates = [
        data_dir / "expiration_management.xlsx",
        repo_root / "data" / "expiration_management.xlsx",
        repo_root
        / "Gategroup 2025-20251025T180038Z-1-001"
        / "Gategroup 2025"
        / "HackMTY2025_ChallengeDimensions"
        / "01_ExpirationDateManagement"
        / "[HackMTY2025]_ExpirationDateManagement_Dataset_v1.xlsx",
    ]
    return candidates


def _load_consumption_summary() -> Dict[str, dict]:
    """Load consumption_prediction.xlsx and return a map by Product_ID with aggregated values.

    Returns dict: { product_id: {standard_spec_qty, unit_cost, quantity_consumed} }
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "data" / "external" / "consumption_prediction.xlsx",
        repo_root / "data" / "consumption_prediction.xlsx",
        repo_root
        / "Gategroup 2025-20251025T180038Z-1-001"
        / "Gategroup 2025"
        / "HackMTY2025_ChallengeDimensions"
        / "02_ConsumptionPrediction"
        / "[HackMTY2025]_ConsumptionPrediction_Dataset_v1.xlsx",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_excel(path)
            except Exception:
                continue

            if "Product_ID" not in df.columns:
                return {}

            agg = (
                df.groupby("Product_ID").agg(
                    standard_spec_qty=("Standard_Specification_Qty", "mean"),
                    unit_cost=("Unit_Cost", "mean"),
                    quantity_consumed=("Quantity_Consumed", "mean"),
                )
            )
            out: Dict[str, dict] = {}
            for pid, row in agg.reset_index().iterrows():
                # pid is integer index, but we want product_id value from row
                prod = row.get("Product_ID") if "Product_ID" in row else None
            # correct iteration
            out = {}
            for _, r in agg.reset_index().iterrows():
                pid = r.get("Product_ID")
                if pd.isna(pid):
                    continue
                out[str(pid)] = {
                    "standard_spec_qty": float(r.get("standard_spec_qty")) if pd.notna(r.get("standard_spec_qty")) else None,
                    "unit_cost": float(r.get("unit_cost")) if pd.notna(r.get("unit_cost")) else None,
                    "quantity_consumed": float(r.get("quantity_consumed")) if pd.notna(r.get("quantity_consumed")) else None,
                }
            return out
    return {}


def _generate_tone_wav_bytes(frequency: float = 880.0, duration: float = 0.5, volume: float = 0.5, sample_rate: int = 22050) -> bytes:
    """Generate a mono WAV (16-bit PCM) with a simple sine tone and return bytes."""
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            t = float(i) / sample_rate
            sample = volume * math.sin(2 * math.pi * frequency * t)
            # clip and convert to int16
            val = max(-1.0, min(1.0, sample))
            packed = struct.pack("<h", int(val * 32767))
            wf.writeframes(packed)
    return buf.getvalue()


def _load_expiration_lots() -> List[dict]:
    """Read expiration_management.xlsx if available and return a list of lot dicts.

    The returned dicts follow the frontend shape used by `LotsPage`.
    """
    candidates = _expiration_paths()
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_excel(path)
            except Exception:
                continue

            lots = []
            # normalize column names to lower-case for flexible mapping
            cols = {c.lower(): c for c in df.columns}

            def colval(row, *names, default=None):
                for n in names:
                    key = n.lower()
                    if key in cols:
                        return row[cols[key]]
                return default

            # try to enrich with consumption summary when available
            consumption_map = _load_consumption_summary()

            for _, row in df.iterrows():
                # build lot dict with safe lookups
                product_id = colval(row, "Product_ID", "product_id", "Product Id")
                product_name = colval(row, "Product_Name", "product_name")
                lot_number = colval(row, "Lot_Number", "lot_number", "Lot")
                expiry = colval(row, "Expiration_Date", "Expiry_Date", "expiry_date", "Expiration")
                origin = colval(row, "Origin", "origin")
                standard_spec = colval(row, "Standard_Specification_Qty", "Standard_Specification_Qty")
                unit_cost = colval(row, "Unit_Cost", "unit_cost", "Unit Cost")
                crew_feedback = colval(row, "Crew_Feedback", "crew_feedback")

                if pd.isna(product_id) and pd.isna(product_name):
                    continue

                # normalize expiry to ISO date string if possible
                expiry_str = None
                if pd.notna(expiry):
                    try:
                        expiry_dt = pd.to_datetime(expiry)
                        expiry_str = expiry_dt.date().isoformat()
                    except Exception:
                        expiry_str = str(expiry)

                # if product id present, try to enrich
                p_id = str(product_id) if pd.notna(product_id) else None
                enrich = consumption_map.get(p_id, {}) if p_id else {}

                lot = {
                    "product_id": str(product_id) if pd.notna(product_id) else None,
                    "product_name": str(product_name) if pd.notna(product_name) else None,
                    "lot_number": str(lot_number) if pd.notna(lot_number) else None,
                    "expiry_date": expiry_str,
                    "origin": str(origin) if pd.notna(origin) else None,
                    "standard_spec_qty": (
                        float(standard_spec) if (pd.notna(standard_spec) and standard_spec != "") else enrich.get("standard_spec_qty")
                    ),
                    "unit_cost": (
                        float(unit_cost) if (pd.notna(unit_cost) and unit_cost != "") else enrich.get("unit_cost")
                    ),
                    "quantity_consumed": enrich.get("quantity_consumed"),
                    "quantity_consumed": None,
                    "crew_feedback": str(crew_feedback) if pd.notna(crew_feedback) else None,
                    "recommended": False,
                }
                lots.append(lot)

            return lots

    # no file found or couldn't parse
    return []


# Load once at module import for demo (uvicorn --reload will reload on changes)
EXPIRATION_LOTS = _load_expiration_lots()


@app.get("/flights")
def flights():
    return JSONResponse(content={"flights": FLIGHTS})


@app.get("/lots/recommend")
def lots_recommend(flight_id: str):
    """Return lot recommendations for a given flight_id.

    If expiration dataset is available, return lots filtered by flight origin.
    Otherwise, return a small fallback sample.
    """
    # find flight origin from static list
    origin = None
    for f in FLIGHTS:
        if f.get("flight_id") == flight_id:
            origin = f.get("origin")
            break

    if EXPIRATION_LOTS:
        if origin:
            filtered = [l for l in EXPIRATION_LOTS if (l.get("origin") is None or str(l.get("origin")).upper() == str(origin).upper())]
        else:
            filtered = EXPIRATION_LOTS

        # mark a couple as recommended heuristically (earliest expiry)
        try:
            with_dates = [l for l in filtered if l.get("expiry_date")]
            with_dates_sorted = sorted(with_dates, key=lambda x: x.get("expiry_date") or "9999-12-31")
            for i, lot in enumerate(with_dates_sorted[:2]):
                lot["recommended"] = True
        except Exception:
            pass

        return JSONResponse(content={"flight_id": flight_id, "origin": origin or "", "lots": filtered})

    # fallback sample when no expiration dataset available
    lots = [
        {"product_id": "P-CHIPS-50", "product_name": "Chips Clásicas 50g", "lot_number": "B47-2025", "expiry_date": "2025-11-12", "standard_spec_qty": 12.0, "quantity_consumed": 8.0, "unit_cost": 6.5, "recommended": True},
        {"product_id": "P-EBAR-60", "product_name": "Energy Bar 60g", "lot_number": "C12-2025", "expiry_date": "2025-11-18", "standard_spec_qty": 10.0, "quantity_consumed": 6.0, "unit_cost": 12.0, "recommended": False},
    ]
    return JSONResponse(content={"flight_id": flight_id, "origin": origin or "MEX", "lots": lots})


@app.post("/predict/financial-impact")
def predict_financial_impact(body: dict):
    # simple deterministic demo calculation based on waste multiplier
    waste_multiplier = body.get("waste_cost_multiplier", 1.0)
    base_waste = 1200000
    base_fuel = 300000
    recovered = 860000
    waste_spir = base_waste * 0.6
    waste_savings = (base_waste - waste_spir) * waste_multiplier
    fuel_savings = base_fuel * 0.12
    total = waste_savings + fuel_savings + recovered
    details = []
    if body.get("include_details"):
        for i in range(min(body.get("max_details", 4), 4)):
            details.append({
                "flight_id": f"AM10{i}",
                "product_id": f"P-00{i}",
                "product_name": f"Producto {i}",
                "unit_cost": 10 + i,
                "recommended_load": 20 - i,
                "baseline_returns": 5 + i,
                "spir_returns": 2 + i,
            })

    return JSONResponse(
        content={
            "assumptions": body,
            "waste_cost_baseline": base_waste,
            "waste_cost_spir": waste_spir,
            "waste_savings": waste_savings,
            "fuel_weight_reduction_kg": 3400,
            "fuel_cost_savings": fuel_savings,
            "recovered_retail_value": recovered,
            "total_impact": total,
            "details": details,
        }
    )



@app.post("/assist/speak")
def assist_speak(body: dict):
    """Demo TTS endpoint: returns a short confirmation tone as WAV bytes for demo purposes.

    The frontend expects an audio blob to play. For the demo we return a short tone.
    """
    text = (body or {}).get("text", "")
    # For demo: ignore text content and return a short speaking-style tone
    wav = _generate_tone_wav_bytes(frequency=650.0, duration=0.6, volume=0.6)
    return Response(content=wav, media_type="audio/wav")


@app.post("/assist/sound")
def assist_sound(body: dict):
    """Demo sound effects endpoint: returns a short chime WAV blob.

    Accepts {prompt: str, duration_seconds: float}
    """
    prompt = (body or {}).get("prompt", "")
    duration = float((body or {}).get("duration_seconds", 0.5) or 0.5)
    # map prompt to frequency heuristically
    freq = 880.0 if "success" in (prompt or "").lower() or "confirm" in (prompt or "").lower() else 660.0
    wav = _generate_tone_wav_bytes(frequency=freq, duration=max(0.15, min(2.0, duration)), volume=0.65)
    return Response(content=wav, media_type="audio/wav")
