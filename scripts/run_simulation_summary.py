import json
import sys
from pathlib import Path
from textwrap import indent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.app.main import app

DEFAULT_PAYLOAD = {
    "date": "2025-06-02",
    "scenario_id": "demo-prueba",
    "plants": [
        {
            "plant_id": 1,
            "name": "Plant 1",
            "base_quantity": 450,
            "staff_baseline": 50,
            "historical_avg_passengers": 2500,
            "historical_avg_flights": 80,
        },
        {
            "plant_id": 2,
            "name": "Plant 2",
            "staff_baseline": 38,
            "historical_avg_flights": 30,
        },
    ],
}


def main() -> None:
    client = TestClient(app)
    response = client.post("/simulate/generative", json=DEFAULT_PAYLOAD)
    if response.status_code != 200:
        print(f"Error al invocar la simulacion: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(response.text)
        return

    data = response.json()
    print(f"Fecha del escenario: {data.get('date')}")
    print(f"ID de escenario: {data.get('scenario_id')}")
    print("")

    for result in data.get("results", []):
        print(f"Planta {result.get('plant_id')}: {result.get('name') or 'Sin nombre'}")

        gemini_payload = result.get("gemini_payload", {}).get("normalized", {})
        print("  Datos generados por Gemini:")
        print(
            indent(
                json.dumps(gemini_payload, ensure_ascii=False, indent=2),
                prefix="    ",
            )
        )

        demand = result.get("demand_prediction") or {}
        staffing = result.get("staffing_prediction") or {}

        print("  Modelo SPIR:")
        print(
            indent(
                json.dumps(
                    {
                        "recommended_quantity": demand.get("recommended_quantity"),
                        "demand_index": demand.get("demand_index"),
                        "recommended_staff": staffing.get("recommended_staff"),
                        "workload_index": staffing.get("workload_index"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                prefix="    ",
            )
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
