[![Backend Smoke](https://github.com/Edson-Zepeda/UdeCodes/actions/workflows/backend-smoke.yml/badge.svg)](https://github.com/Edson-Zepeda/UdeCodes/actions/workflows/backend-smoke.yml)

# SPIR - Kit de Planificación Inteligente (HackMTY 2025)

SPIR (Smart Planning Intelligent Resources) es la base de datos y backend de nuestra solución para HackMTY 2025 con gategroup. Este repositorio incluye:

- Un notebook de extremo a extremo que entrena un modelo de consumo con MAPE < 2%.
- Artefactos reproducibles (.pkl) listos para servir.
- Un backend FastAPI que expone simuladores de dotación, demanda e impacto financiero.

> Estado: El frontend se trabaja por separado; esta rama contiene el stack completo de datos y backend para reproducir resultados e interactuar con las APIs.

---

## Estructura del Repositorio

```
backend/
  app/
    financial.py          # motor de impacto financiero (modelo + métricas)
    main.py               # punto de entrada de FastAPI
    models.py             # esquemas Pydantic de entrada/salida
docs/
  images/banner.png
notebooks/
  consumption_prediction.ipynb
data/
  external/               # coloca aquí los datasets Excel (ver abajo)
scripts/
  ...                     # scripts auxiliares opcionales
```

Los artefactos generados por el notebook (por ejemplo, `backend/models/consumption_prediction_xgb.pkl`) se comitean para que el backend pueda correr sin reentrenar.

---

## 1. Preparación y Dependencias

1. Python 3.11 (recomendado)
   ```
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows PowerShell
   source .venv/bin/activate   # macOS/Linux
   ```

2. Instalar dependencias del proyecto
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Colocar datos: copia los datasets oficiales de HackMTY a `data/external/`.

   | Archivo requerido | Fuente |
   | ----------------- | ------ |
   | `consumption_prediction.xlsx` | `[HackMTY2025]_ConsumptionPrediction_Dataset_v1.xlsx` |
   | `expiration_management.xlsx`  | `[HackMTY2025]_ExpirationDateManagement_Dataset_v1.xlsx` |
   | `productivity_estimation.xlsx`| `[HackMTY2025]_ProductivityEstimation_Dataset_v1.xlsx` |

   Usa exactamente esos nombres de archivo; los loaders del backend se encargan del resto.

4. Acceso a Gemini (opcional)

   Configura tu API key de Gemini:

   ```powershell
   setx GENAI_API_KEY "<tu-key>"   # Windows (persistente)
   ```

   ```bash
   export GENAI_API_KEY="<tu-key>" # macOS / Linux
   ```

   Para demos sin conexión o si aún no tienes key:

   ```powershell
   $env:GEMINI_OFFLINE = "1"
   ```

   Con `GEMINI_OFFLINE=1` el backend usa escenarios sintéticos deterministas para que todos los endpoints funcionen sin llamadas externas.

---

## 2. Reproducir el Notebook de Consumo

1. Regenerar el scaffold del notebook (opcional)
   ```
   python create_consumption_notebook.py
   ```

2. Ejecutarlo de forma no interactiva
   ```
   python -m nbconvert --to notebook --execute --inplace notebooks/consumption_prediction.ipynb
   ```

   Este paso actualiza:
   - `backend/models/consumption_prediction_xgb.pkl`
   - métricas de evaluación (impresas dentro del notebook)
   - tablas de monitoreo (MAPE por producto, etc.)

3. Verificación rápida de métricas (opcional)
   ```
   python -c "from backend.app.financial import calculate_financial_impact; \
              r=calculate_financial_impact(); \
              print(f'Test MAPE <= 2% logrado, impacto total ${r.total_impact:,.2f}')"
   ```

---

## 3. Ejecutar la API del Backend

1. Asegúrate de tener el entorno virtual activo y el modelo `.pkl` en `backend/models/`.

2. Inicia FastAPI:
   ```
   uvicorn backend.app.main:app --reload
   ```

3. Abre `http://127.0.0.1:8000/docs` para la documentación interactiva (Swagger).

### Endpoints expuestos

| Endpoint | Propósito | Ejemplo de payload |
| -------- | --------- | ------------------ |
| `POST /predict/demand` | IDD de pasajeros y recomendación de cantidad | `{"date":"2025-06-02","plant_id":1,"base_quantity":450}` |
| `POST /predict/staffing` | Índice de carga de vuelos y dotación recomendada | `{"date":"2025-06-02","plant_id":1,"staff_baseline":50}` |
| `POST /predict/financial-impact` | Métricas financieras consolidadas | ver siguiente sección |
| `POST /simulate/what-if` | Escenario impulsado por Gemini + delta financiero | ver ejemplo abajo |
| `POST /assist/speak` | Streaming de audio TTS de ElevenLabs (mp3) | `{"text":"Carga 15 aguas al vuelo AM109"}` |
| `POST /assist/sound` | Efectos de sonido ElevenLabs para la UI | `{"prompt":"positive confirmation chime"}` |

---

## 4. API del Simulador de Impacto Financiero

Este endpoint alimenta los controles del Simulador de Impacto Financiero de SPIR.

```bash
curl -X POST http://127.0.0.1:8000/predict/financial-impact \
  -H "Content-Type: application/json" \
  -d '{
        "fuel_cost_per_liter": 28.0,
        "waste_cost_multiplier": 1.10,
        "unit_margin_factor": 3.0,
        "include_details": true,
        "max_details": 10
      }'
```

Campos de respuesta:
- `waste_cost_baseline`, `waste_cost_spir`, `waste_savings`
- `fuel_weight_reduction_kg`, `fuel_cost_savings`
- `recovered_retail_value`
- `total_impact`
- Opcional `details[]` (limitado por `max_details`) con insights por vuelo/producto para la tabla del dashboard.

#### Endpoint de escenario "What-If"

```bash
curl -X POST http://127.0.0.1:8000/simulate/what-if \
  -H "Content-Type: application/json" \
  -d '{
        "scenario": "Simula un pico de demanda por feriado en la Planta 3",
        "date": "2025-12-20",
        "plants": [
          {
            "plant_id": 3,
            "historical_avg_passengers": 2500,
            "historical_avg_flights": 80,
            "base_quantity": 450,
            "staff_baseline": 50
          }
        ],
        "financial_assumptions": {
          "fuel_cost_per_liter": 28.0,
          "waste_cost_multiplier": 1.10,
          "unit_margin_factor": 3.0,
          "include_details": true,
          "max_details": 25
        }
      }'
```

La respuesta incluye los payloads simulados de Gemini, métricas de dotación/demanda refrescadas y un resumen financiero en tres partes (baseline vs. escenario vs. delta). Integra este output con el botón "Simular Escenario" en el dashboard.

Relación de sliders con parámetros del request:

| Control (frontend) | Clave en payload | Efecto |
| ------------------ | ---------------- | ------ |
| Costo promedio por comida desperdiciada | `waste_cost_multiplier` | Escala el costo de merma |
| Precio del combustible por litro | `fuel_cost_per_liter` | Ajusta el ahorro por combustible |
| Margen retail (%) | `unit_margin_factor` | Aumenta el valor recuperado |
| (Opcional) Factor de seguridad | `buffer_factor` | Controla el buffer recomendado de carga |

---

## 5. Flujo de Ramas

| Rama | Estado | Acción siguiente |
| ---- | ------ | ---------------- |
| `main` | estable (commit inicial) | mantener sin cambios hasta el demo |
| `develop` | integración | hacer merge de features completas aquí |
| `feature/consumption-prediction-model` | listo | PR a `develop` (notebook + modelo) |
| `feature/setup-mock-api` | listo | PR a `develop` (tras integrar el modelo) |
| `feature/build-dashboard-ui` | en progreso | rebase en `develop`, conectar API real |
| `feature/integrate-real-model` | planificado | cargar `.pkl`, servir predicciones |
| `feature/financial-dashboard-ui` | planificado | construir la UI del simulador financiero |
| `feature/integrate-all-challenges` | planificado | endpoints para retos restantes |

Resumen del flujo:
1. Hacer merge de las dos ramas completas a `develop`.
2. Crear nuevas ramas desde `develop` actualizado.
3. Tras QA, fast-forward de `develop` -> `main` para el release del demo.

---

## 6. Frontend (Vite + React)

1. Instala Node.js 20 o superior.
2. Prepara dependencias:
   ```bash
   cd frontend
   npm install
   ```
3. Configura `.env` dentro de `frontend/` con la URL del backend:
   ```
   VITE_API_BASE_URL=http://127.0.0.1:8000
   ```
4. Corre el servidor de desarrollo:
   ```bash
   npm run dev
   ```
   Abre `http://127.0.0.1:5173` y valida las tres vistas (Inicio, Lotes, Dashboard).

El dashboard activa por defecto el asistente de voz. Para desactivar el audio, usa el toggle "Asistente de voz" en la esquina superior derecha.

---

## 7. Lista de Validación

- [x] El notebook ejecuta sin ediciones manuales (compatible con `nbconvert`).
- [x] Artefactos del modelo en `backend/models/`.
- [x] El backend inicia con el modelo entrenado y devuelve métricas reales.
- [x] `/predict/financial-impact` maneja sliders y tablas de detalle opcionales.
- [ ] Frontend (repo/rama separados) consume los endpoints - pendiente de integración.

---

## 8. Comandos Útiles

```
# Lint / formato (agrega tus herramientas preferidas, p. ej. ruff o black)

# Ejecutar backend local
uvicorn backend.app.main:app --reload

# Prueba de humo de la API
python - <<'PY'
from fastapi.testclient import TestClient
from backend.app.main import app
client = TestClient(app)
assert client.get("/health").json()["status"] == "ok"
assert client.post("/predict/financial-impact", json={}).status_code == 200
print("Smoke tests passed.")
PY
```

---

### Preguntas

Los notebooks, scripts y endpoints de API están contenidos en este repositorio. Si necesitas más contexto (por ejemplo, wiring del frontend o retos adicionales), revisa el plan de ramas anterior o contáctanos en el equipo de datos/backend de SPIR.

