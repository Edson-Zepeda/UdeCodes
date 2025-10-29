"""Utilities to talk with the Gemini API and provide synthetic fallbacks."""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai

logger = logging.getLogger(__name__)

DEFAULT_API_KEY = ""
API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("API_KEY") or DEFAULT_API_KEY
_CLIENT: Optional[genai.Client] = None

MASTER_PROMPT_TEMPLATE = (
    "Actua como simulador de operaciones aereas para un equipo de catering.\n"
    "Genera un unico objeto JSON con el formato indicado y sin texto adicional.\n\n"
    "Formato:\n"
    '- Plantas 1 y 3: {"day": YYYYMMDD, "flights": int, "passengers": int, "max capacity": int}\n'
    '- Plantas 2,4,5,6: {"day": YYYYMMDD, "flights": int}\n\n'
    "Reglas:\n"
    "- passengers < max capacity\n"
    "- max capacity ~= passengers * (1.05 .. 1.15)\n\n"
    "Genera los datos para la planta {plant_id} y la fecha {flight_date}.\n"
    "La salida debe ser unicamente el objeto JSON."
)

SCENARIO_PROMPT_TEMPLATE = (
    "Actua como consultor de planeacion para gategroup.\n"
    "Escenario descrito por el gerente:\n"
    '\"\"\"{scenario_text}\"\"\"\n\n'
    "Instrucciones:\n"
    '- Genera un unico objeto JSON.\n'
    '- Incluye las claves: "plant_id", "day" (YYYYMMDD), "flights", "passengers" y "max capacity".\n'
    "- Ajusta los valores usando el escenario y los datos base.\n"
    "- Respeta: passengers < max capacity y max capacity ~= passengers * (1.05 .. 1.15).\n\n"
    "Datos base de la planta {plant_id}:\n"
    "{plant_context}"
)

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "flights.db"

SYNTHETIC_BASELINES: Dict[int, Dict[str, Any]] = {
    1: {
        "flights": [352, 356, 358, 365, 349],
        "passengers_per_flight": (250.0, 28.0),
    },
    2: {"flights": [19, 23, 20, 21, 23, 22]},
    3: {
        "flights": [118, 124, 131, 129, 135],
        "passengers_per_flight": (220.0, 24.0),
    },
    4: {"flights": [42, 38, 44, 47, 39]},
    5: {"flights": [16, 19, 17, 21, 18]},
    6: {"flights": [55, 61, 58, 64, 59]},
}


def _normalize_date_str(value: str) -> str:
    value = str(value).strip()
    if value.isdigit() and len(value) == 8:
        return value
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return value.replace("-", "").replace("/", "")


def _get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not API_KEY:
        raise RuntimeError(
            "Falta la API key. Define la variable de entorno GENAI_API_KEY o API_KEY."
        )
    _CLIENT = genai.Client(api_key=API_KEY)
    return _CLIENT


def _extract_text_from_response(response: Any) -> str:
    for attr in ("text", "content", "output", "body"):
        if hasattr(response, attr):
            value = getattr(response, attr)
            if isinstance(value, str):
                return value.strip()
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                continue
    return str(response)


def _find_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cur.fetchall()]


def _get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    return [row[1] for row in cur.fetchall()]


def _fetch_aggregated_from_db(plant_id: int, flight_date: str) -> Optional[Dict[str, Any]]:
    if not DB_PATH.exists():
        return None

    normalized_date = flight_date.replace("-", "")
    with sqlite3.connect(DB_PATH) as conn:
        tables = _find_tables(conn)
        for table in tables:
            cols = _get_columns(conn, table)
            plant_columns = [c for c in cols if c.lower() in {"plant_id", "plant", "plantid"}]
            date_columns = [c for c in cols if c.lower() in {"date", "day", "flight_date", "fecha"}]
            if not plant_columns or not date_columns:
                continue

            plant_col = plant_columns[0]
            date_col = date_columns[0]
            cur = conn.cursor()
            try:
                cur.execute(
                    f"""
                    SELECT *
                    FROM '{table}'
                    WHERE {plant_col} = ?
                      AND ({date_col} = ? OR {date_col} = ?)
                    """,
                    (plant_id, flight_date, normalized_date),
                )
            except sqlite3.OperationalError:
                continue

            rows = cur.fetchall()
            if not rows:
                continue

            def resolve_column(candidates: list[str]) -> Optional[str]:
                for candidate in candidates:
                    for column in cols:
                        if column.lower() == candidate:
                            return column
                return None

            flights_column = resolve_column(["flights", "flight_count", "n_flights", "num_flights"])
            passengers_column = resolve_column(["passengers", "pax", "passenger_count"])
            capacity_column = resolve_column(["max_capacity", "capacity", "max capacity", "maxcap"])

            result: Dict[str, Any] = {"day": int(normalized_date)}
            if flights_column:
                flights = sum((row[cols.index(flights_column)] or 0) for row in rows)
                result["flights"] = int(flights)
            else:
                result["flights"] = len(rows)

            if passengers_column:
                passengers = sum((row[cols.index(passengers_column)] or 0) for row in rows)
                result["passengers"] = int(passengers)

            if capacity_column:
                capacities = [row[cols.index(capacity_column)] or 0 for row in rows]
                result["max capacity"] = int(max(capacities))

            logger.info("Datos agregados desde DB tabla %s: %s", table, result)
            return result

    return None


def build_prompt(plant_id: int, flight_date: str) -> str:
    formatted_date = flight_date.replace("-", "")
    return MASTER_PROMPT_TEMPLATE.format(plant_id=plant_id, flight_date=formatted_date)


def generate_flight_data(
    plant_id: int,
    flight_date: str,
    prompt: Optional[str] = None,
    skip_db: bool = False,
) -> Optional[Dict[str, Any]]:
    if not skip_db:
        db_row = _fetch_aggregated_from_db(plant_id, flight_date)
        if db_row:
            return db_row

    offline_mode = os.getenv("GEMINI_OFFLINE") == "1" or API_KEY == DEFAULT_API_KEY
    if offline_mode:
        fallback = generate_example_row(plant_id, flight_date)
        if fallback:
            fallback.setdefault("day", _normalize_date_str(flight_date))
        return fallback

    prompt = prompt or build_prompt(plant_id, flight_date)

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        payload = json.loads(_extract_text_from_response(response))
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.error("No se pudieron generar datos con Gemini: %s", exc)
        return None


def build_scenario_prompt(
    plant_id: int,
    scenario_text: str,
    flight_date: str,
    hints: Optional[Dict[str, Any]] = None,
) -> str:
    hints = hints or {}
    lines = [f"- Fecha simulada: {flight_date}"]
    if hints.get("name"):
        lines.append(f"- Nombre referencial: {hints['name']}")
    if hints.get("baseline_passengers"):
        lines.append(f"- Promedio historico de pasajeros: {hints['baseline_passengers']}")
    if hints.get("baseline_flights"):
        lines.append(f"- Promedio historico de vuelos: {hints['baseline_flights']}")
    if hints.get("notes"):
        lines.append(f"- Notas adicionales: {hints['notes']}")
    context = "\n".join(lines) if lines else "Sin datos adicionales."
    return SCENARIO_PROMPT_TEMPLATE.format(
        scenario_text=scenario_text,
        plant_id=plant_id,
        plant_context=context,
    )


def _sample_normal(values: list[float]) -> tuple[float, float]:
    mean_val = statistics.mean(values)
    if len(values) > 1:
        std_val = statistics.stdev(values)
    else:
        std_val = max(1.0, mean_val * 0.05)
    return mean_val, max(std_val, 1.0)


def generate_example_row(
    plant_id: int,
    flight_date: str,
    rnd_seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    baseline = SYNTHETIC_BASELINES.get(int(plant_id))
    if not baseline:
        return None

    if rnd_seed is not None:
        random.seed(rnd_seed)

    flights_stats = _sample_normal([float(v) for v in baseline["flights"]])
    flights = max(1, int(round(random.gauss(*flights_stats))))

    row: Dict[str, Any] = {"day": int(_normalize_date_str(flight_date)), "flights": flights}

    pax_stats = baseline.get("passengers_per_flight")
    if pax_stats:
        passengers_per_flight = max(50, min(int(round(random.gauss(*pax_stats))), 450))
        passengers = passengers_per_flight * flights
        capacity_multiplier = random.uniform(1.05, 1.12)
        max_capacity = max(passengers + flights * 8, int(round(passengers * capacity_multiplier)))
        row["passengers"] = int(passengers)
        row["max capacity"] = int(max_capacity)

    logger.info("Fila sintetica generada para planta %s: %s", plant_id, row)
    return row


def prepare_api_payload(row: Dict[str, Any], plant_id: int) -> Dict[str, Any]:
    if not row:
        return {}

    payload: Dict[str, Any] = {
        "plant_id": int(plant_id),
        "day": _normalize_date_str(row.get("day", "")),
        "flights": int(row.get("flights", 0)),
    }

    if "passengers" in row and row["passengers"] is not None:
        payload["passengers"] = int(row["passengers"])
    elif "pax" in row and row["pax"] is not None:
        payload["passengers"] = int(row["pax"])

    if "max capacity" in row and row["max capacity"] is not None:
        payload["max_capacity"] = int(row["max capacity"])
    elif "max_capacity" in row and row["max_capacity"] is not None:
        payload["max_capacity"] = int(row["max_capacity"])
    elif "capacity" in row and row["capacity"] is not None:
        payload["max_capacity"] = int(row["capacity"])

    return payload


def payload_to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def send_payload_to_api(
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        return {"status_code": response.status_code, "text": response.text}


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: python gemini.py <plant_id> <date> [seed]", file=sys.stderr)
        return 1
    plant_id = int(argv[1])
    flight_date = argv[2]
    seed = int(argv[3]) if len(argv) > 3 else None
    row = generate_example_row(plant_id, flight_date, rnd_seed=seed)
    if not row:
        print(f"Sin datos sinteticos para la planta {plant_id}", file=sys.stderr)
        return 2
    payload = prepare_api_payload(row, plant_id)
    print(payload_to_json(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
