# Resumen de Validación del Problema: gategroup

Este documento resume hallazgos del Análisis Exploratorio de Datos (EDA) sobre los datasets proporcionados por gategroup.

## 1. El Problema

Tras analizar plantas 1 y 2 observamos:

- Alta volatilidad diaria en la carga de trabajo (vuelos y pasajeros).
- Datos agregados a nivel diario que dificultan decisiones tácticas.

La combinación de variaciones abruptas (picos en fines de semana o festivos) y escasa granularidad complica la planificación táctica y deriva en:

- Desperdicio elevado: vuelos regresan con >50% de artículos sin usar.
- Productividad inconsistente: asignación reactiva de personal y cuellos de botella (3.5–7 min por carrito).

> Gráfico de ejemplo (Plant 1.csv): volatilidad diaria de pasajeros (indicador clave de imprevisibilidad).

## 2. La Oportunidad

Aun cuando hoy se planifica de forma reactiva, la carga de trabajo no es aleatoria. Existen patrones temporales (semanales/mensuales/estacionales) que pueden modelarse y anticiparse con alta precisión.

Proponemos transformar datos operativos “macro” en inteligencia “micro” accionable: un pronóstico diario sólido permite modular listas de empaque y personal, conectando planificación estratégica con la ejecución en planta.

## 3. Solución Propuesta (SPIR)

Sistema de Planificación Inteligente de Recursos (SPIR):

- Modelo de forecasting (XGBoost) para predecir vuelos/pasajeros con error <2%.
- Índice de Demanda Diaria (IDD) para guiar decisiones operativas.
  - Consumo: ajustar dinámicamente cantidades cargadas por vuelo (menos desperdicio y quiebres).
  - Productividad: estimar personal diario recomendado estabilizando la operación.

