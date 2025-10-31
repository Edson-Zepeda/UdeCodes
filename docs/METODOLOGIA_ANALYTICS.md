# Metodología de Análisis y Especificación del Modelo

Este documento describe lógica, fórmulas y contratos de datos (I/O) para los motores de predicción y optimización del proyecto.

## 1. Objetivo del Modelo

Automatizar decisiones que impactan beneficio y servicio en planta mediante tres capacidades:

- Optimización de Lotes (FEFO): rotación óptima para reducir mermas por caducidad.
- Predicción de Demanda (IDD): ajustar cantidades cargadas según demanda pronosticada.
- Productividad: estimar personal necesario según carga de trabajo pronosticada.

## 2. Modelos y Fórmulas

### 2.1 Selección de Lote Óptimo (FEFO)

Regla de negocio: First-Expired, First-Out.

Pseudocódigo:

```
funcion obtenerLoteOptimo(lotes, fecha_vuelo):
  lotes_validos = filtrar(lotes, lote.fecha_caducidad > fecha_vuelo)
  lotes_ordenados = ordenar(lotes_validos, por=fecha_caducidad ASC)
  return primer(lotes_ordenados)
```

### 2.2 Predicción de Cantidad (Índice de Demanda Diaria, IDD)

Transforma la “Spec Base” estática en una lista dinámica por vuelo:

Q_final = ceil((P_pronosticados / P_histórico) * Q_base)

Donde:

- Q_final: cantidad final a surtir (entero).
- P_pronosticados: pasajeros pronosticados por el modelo.
- P_histórico: promedio histórico de pasajeros.
- Q_base: cantidad estándar de la ruta.

### 2.3 Estimación de Personal Necesario

E_necesarios = ceil(E_base * (V_pronosticados / V_histórico))

Donde:

- E_necesarios: número recomendado de empleados.
- E_base: plantilla base para un día promedio.
- V_pronosticados: vuelos pronosticados por el modelo.
- V_histórico: promedio histórico de vuelos.

## 3. Contrato de Datos (API)

- POST /predict/demand → demanda/IDD de pasajeros.
- POST /predict/staffing → índice de carga de vuelos y personal sugerido.
- POST /predict/financial-impact → métricas financieras agregadas y detalles opcionales.

## 4. Entrenamiento y Artefactos

- Notebook de consumo genera `backend/models/consumption_prediction_xgb.pkl`.
- Sustituible vía variable `MODEL_URL` en despliegues (descarga automática en arranque).

## 5. Validación

- Métrica objetivo: MAPE <2% en consumo a nivel agregado.
- Pruebas de humo: `/health`, `/predict/financial-impact` → 200 con payload válido aun sin artefacto local.

