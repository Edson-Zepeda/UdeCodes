import os
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Any, Dict
from google import genai

logging.basicConfig(level=logging.INFO)
API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("API_KEY")
client = None  # inicialización perezosa

MASTER_PROMPT_TEMPLATE = """
Actua como simulador de datos de operaciones aereas para una empresa de catering.
Genera un unico objeto JSON valido con el formato indicado y sin texto adicional.

Formato:
- Plantas 1 y 3: {{"day": YYYYMMDD, "flights": int, "passengers": int, "max capacity": int}}
- Plantas 2,4,5,6: {{"day": YYYYMMDD, "flights": int}}

Reglas:
- passengers < max capacity
- max capacity ~ passengers * (1.05 .. 1.15)

Genera los datos para la Planta {plant_id} y la fecha {flight_date}.
La salida debe ser unicamente el objeto JSON.
""".strip()

def _get_client():
    global client
    if client is not None:
        return client
    key = os.getenv("GENAI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("Falta la API key. Define la variable de entorno GENAI_API_KEY o API_KEY.")
    client = genai.Client(api_key=key)
    return client

ROOT = Path.cwd()
DB_PATH = ROOT / "flights.db"

def _extract_text_from_response(response: Any) -> str:
    for attr in ("text", "content", "output", "body"):
        if hasattr(response, attr):
            value = getattr(response, attr)
            if isinstance(value, str):
                return value.strip()
            try:
                return json.dumps(value)
            except Exception:
                continue
    return str(response)

def _find_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]

def _get_columns(conn: sqlite3.Connection, table: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    return [row[1] for row in cur.fetchall()]

def _fetch_aggregated_from_db(plant_id: int, flight_date: str) -> Optional[Dict]:
    if not DB_PATH.exists():
        logging.info("No existe flights.db en %s", DB_PATH)
        return None

    normalized_date = flight_date.replace('-', '')
    with sqlite3.connect(DB_PATH) as conn:
        tables = _find_tables(conn)
        for t in tables:
            cols = _get_columns(conn, t)
            # detectar columnas candidatas
            plant_cols = [c for c in cols if c.lower() in ("plant_id", "plant", "plantid")]
            date_cols = [c for c in cols if c.lower() in ("date", "day", "flight_date", "fecha")]
            if not plant_cols or not date_cols:
                continue

            plant_col = plant_cols[0]
            date_col = date_cols[0]
            cur = conn.cursor()
            try:
                cur.execute(
                    f"SELECT * FROM '{t}' WHERE {plant_col} = ? AND ({date_col} = ? OR {date_col} = ?)",
                    (plant_id, flight_date, normalized_date)
                )
            except sqlite3.OperationalError:
                continue

            rows = cur.fetchall()
            if not rows:
                continue

            # nombres de columnas
            col_names = cols
            def find_col(candidates):
                for cand in candidates:
                    for c in col_names:
                        if c.lower() == cand:
                            return c
                return None

            flights_col = find_col(['flights', 'flight_count', 'n_flights', 'num_flights'])
            pax_col = find_col(['passengers', 'pax', 'passenger_count'])
            cap_col = find_col(['max_capacity', 'capacity', 'max capacity', 'maxcap'])

            # agregados
            if flights_col:
                flights = sum((row[col_names.index(flights_col)] or 0) for row in rows)
            else:
                flights = len(rows)

            passengers = None
            if pax_col:
                passengers = sum((row[col_names.index(pax_col)] or 0) for row in rows)

            max_capacity = None
            if cap_col:
                # tomar el máximo si hay varios registros
                max_capacity = max((row[col_names.index(cap_col)] or 0) for row in rows)

            result = {"day": int(normalized_date), "flights": int(flights)}
            if passengers is not None:
                result["passengers"] = int(passengers)
            if max_capacity is not None:
                result["max capacity"] = int(max_capacity)

            logging.info("Datos obtenidos desde DB tabla %s: %s", t, result)
            return result

    return None

def build_prompt(plant_id: int, flight_date: str) -> str:
    formatted_date = flight_date.replace('-', '')
    return MASTER_PROMPT_TEMPLATE.format(plant_id=plant_id, flight_date=formatted_date)


def generate_flight_data(
    plant_id: int,
    flight_date: str,
    prompt: Optional[str] = None,
) -> Optional[dict]:
    # 1) Intentar obtener desde flights.db
    db_data = _fetch_aggregated_from_db(plant_id, flight_date)
    if db_data:
        return db_data

    # 2) Si no hay datos en la DB, usar Gemini para generar datos sintéticos
    prompt = prompt or build_prompt(plant_id, flight_date)

    try:
        cli = _get_client()
        response = cli.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        json_text = _extract_text_from_response(response)
        data = json.loads(json_text)
        return data
    except Exception as e:
        logging.error("Error al generar datos con Gemini: %s", e)
        return None

def generate_example_row(plant_id: int, flight_date: str, rnd_seed: Optional[int] = None) -> Optional[Dict]:
    """
    Genera UNA fila de ejemplo aleatoria basada en los datos que proporcionaste.
    Produce valores realistas: 'passengers' y 'max capacity' son totales diarios,
    calculados a partir de un promedio por vuelo y con límites por vuelo.
    """
    import random, statistics
    from datetime import datetime

    if rnd_seed is not None:
        random.seed(rnd_seed)

    def _norm_date(s: str) -> int:
        s = str(s).strip()
        if s.isdigit() and len(s) == 8:
            return int(s)
        fmts = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]
        for fmt in fmts:
            try:
                return int(datetime.strptime(s, fmt).strftime("%Y%m%d"))
            except Exception:
                continue
        parts = s.split("/")
        if len(parts) == 3:
            try:
                m, d, y = parts
                return int(datetime(int(y), int(m), int(d)).strftime("%Y%m%d"))
            except Exception:
                pass
        return int(s.replace("-", "").replace("/", ""))

    # ejemplos (tus datos)
    samples = {
        1: {
            "flights": [352, 356, 358, 365],
            "passengers": [94434, 92710, 88888, 88276],
            "max_capacity": [99351, 99616, 99437, 99174],
        },
        2: {
            "flights": [19, 23, 20, 21, 23]
        }
    }

    if plant_id not in samples:
        return None

    # estadísticas por vuelo (cuando aplique)
    def mean_sd_per_flight(pass_list, flight_list):
        if not pass_list or not flight_list:
            return None, None
        per_flight = [p / f if f and f > 0 else 0 for p, f in zip(pass_list, flight_list)]
        mu = statistics.mean(per_flight)
        sd = statistics.stdev(per_flight) if len(per_flight) > 1 else max(1.0, mu * 0.05)
        return mu, max(sd, 1.0)

    # calc stats
    mu_f = statistics.mean(samples[plant_id]["flights"])
    sd_f = statistics.stdev(samples[plant_id]["flights"]) if len(samples[plant_id]["flights"]) > 1 else max(1.0, mu_f * 0.05)

    mu_pf, sd_pf = None, None
    if plant_id == 1:
        mu_pf, sd_pf = mean_sd_per_flight(samples[1]["passengers"], samples[1]["flights"])

    result = {"day": _norm_date(flight_date)}

    # generar flights (entero, no negativo)
    gen_f = max(1, int(round(random.gauss(mu_f, sd_f))))
    result["flights"] = gen_f

    if plant_id == 1:
        # generar pasajeros totales a partir de pasajeros por vuelo, con límites razonables por vuelo
        if mu_pf is None:
            mu_pf = 250.0
            sd_pf = 30.0

        per_flight = int(round(random.gauss(mu_pf, sd_pf)))
        # límites por vuelo para evitar valores absurdos (ej: no más de 450 ni menos de 50 por vuelo)
        per_flight = max(50, min(per_flight, 450))

        total_passengers = max(0, per_flight * gen_f)
        # max capacity total: 5-12% por encima del total de pasajeros o al menos passengers + gen_f*10
        cap_multiplier = random.uniform(1.05, 1.12)
        total_capacity = max(total_passengers + gen_f * 10, int(round(total_passengers * cap_multiplier)))

        result["passengers"] = int(total_passengers)
        result["max capacity"] = int(total_capacity)

    logging.info("Fila de ejemplo generada para planta %s: %s", plant_id, result)
    return result

def prepare_payload(row: Dict, plant_id: int) -> Dict:
    payload = {
        "plant_id": int(plant_id),
        "day": str(row["day"]).zfill(8),
        "flights": int(row["flights"])
    }
    if "passengers" in row:
        payload["passengers"] = int(row["passengers"])
    if "max capacity" in row:
        payload["max_capacity"] = int(row["max capacity"])
    return payload

def prepare_api_payload(row: Dict, plant_id: int) -> Dict:
    """
    Normaliza las claves y crea un payload listo para enviar a una API.
    - row: diccionario devuelto por generate_example_row o _fetch_aggregated_from_db
    - plant_id: entero
    Devuelve un dict con claves seguras: plant_id, day (YYYYMMDD string), flights, passengers (opcional), max_capacity (opcional)
    """
    if row is None:
        return {}

    # normalizar claves posibles
    payload = {
        "plant_id": int(plant_id),
        "day": str(row.get("day", "")).zfill(8),
        "flights": int(row.get("flights", 0))
    }

    # pasajeros
    if "passengers" in row:
        payload["passengers"] = int(row["passengers"])
    elif "pax" in row:
        payload["passengers"] = int(row["pax"])

    # capacidad máxima (usar nombre sin espacios)
    if "max capacity" in row:
        payload["max_capacity"] = int(row["max capacity"])
    elif "max_capacity" in row:
        payload["max_capacity"] = int(row["max_capacity"])
    elif "capacity" in row:
        payload["max_capacity"] = int(row["capacity"])

    return payload

def payload_to_json(payload: Dict) -> str:
    """Convierte el payload a JSON (utf-8 friendly)."""
    return json.dumps(payload, ensure_ascii=False)

def send_payload_to_api(url: str, payload: Dict, api_key: Optional[str] = None, timeout: int = 10) -> Dict:
    """
    Envia el payload a una API (POST). Devuelve response.json() si la respuesta tiene JSON, en caso de error lanza excepción.
    Requiere 'requests' instalado.
    """
    import requests
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {"status_code": resp.status_code, "text": resp.text}

if __name__ == "__main__":
    # uso: python gemini.py <plant_id> <YYYY-MM-DD or 1/2/2023> [seed]
    if len(sys.argv) < 3:
        print("Usage: python gemini.py <plant_id> <date> [seed]", file=sys.stderr)
        sys.exit(1)
    pid = int(sys.argv[1])
    date = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    row = generate_example_row(pid, date, rnd_seed=seed)
    payload = prepare_payload(row, pid)
    print(json.dumps(payload, ensure_ascii=False))

    import json
    # row = generate_example_row(...)  # ya presente en tu módulo

    row = generate_example_row(1, "2023-01-02", rnd_seed=42)

    payload = {
        "plant_id": 1,
        "day": str(row["day"]).zfill(8),
        "flights": int(row["flights"])
    }

    # opcionales
    if "passengers" in row:
        payload["passengers"] = int(row["passengers"])
    if "max capacity" in row:
        payload["max_capacity"] = int(row["max capacity"])

    json_payload = json.dumps(payload, ensure_ascii=False)
    print(json_payload)
