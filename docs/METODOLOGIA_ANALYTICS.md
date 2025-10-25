¡Claro que sí\! Este es un excelente borrador. Lo he reestructurado y pulido para darle un formato más profesional, asegurando que las fórmulas sean claras y que la lógica se alinee perfectamente con la estrategia que hemos discutido, utilizando solo los datos disponibles del reto.

Aquí tienes la versión final del documento.

-----

# Metodología de Análisis y Especificación del Modelo

Este documento describe la lógica central, las fórmulas y los contratos de datos (I/O) para los tres motores de predicción y optimización de nuestro proyecto.

## 1\. Objetivo del Modelo

El objetivo principal es automatizar procesos que impactan directamente el beneficio y prestigio de la empresa. Para ello, nos centramos en resolver tres problemáticas clave del pilar "Smart Intelligence":

1.  **Optimización de Lotes (FEFO):** Asegurar la rotación óptima del inventario para reducir mermas por caducidad y garantizar la calidad del producto, implementando una lógica *First-Expired, First-Out*.
2.  **Predicción de Demanda (Modelo IDD):** Ajustar dinámicamente la cantidad de productos cargados en cada vuelo para que coincida con la demanda real de pasajeros, reduciendo el desperdicio masivo (reportado como \>50%) y los costos de combustible asociados al peso innecesario. [1]
3.  **Optimización de Productividad:** Estimar el personal necesario para la operación diaria basándose en la carga de trabajo pronosticada, permitiendo una planificación de recursos eficiente y evitando los cuellos de botella que generan inconsistencias en la productividad. [1]

-----

## 2\. Las Fórmulas (Los Modelos)

### Modelo 1: Selección de Lote Óptimo (FEFO)

Este es un **algoritmo de selección** que sigue la regla de negocio **FEFO (First-Expired, First-Out)**. Su función es guiar al personal para que siempre utilice los productos con la fecha de caducidad más próxima, minimizando el riesgo de mermas.

El algoritmo primero filtra los lotes ya caducados y luego ordena los restantes para consumir primero los que están más cerca de expirar.

**Lógica del Algoritmo (Pseudocódigo):**

```
// Función para obtener el lote óptimo
FUNCIÓN obtenerLoteÓptimo(lista_de_lotes, fecha_de_vuelo):
  
  // 1. Filtrar lotes no caducados
  lotes_validos = FILTRAR lotes DONDE lote.fecha_caducidad > fecha_de_vuelo
  
  // 2. Ordenar por fecha de caducidad más próxima
  lotes_ordenados = ORDENAR lotes_validos POR lote.fecha_caducidad ASCENDENTE
  
  // 3. Seleccionar el primero de la lista
  RETURN lotes_ordenados
```

### Modelo 2: Predicción de Cantidad de Producto (Modelo IDD)

Este modelo ajusta la cantidad de productos a cargar en un vuelo utilizando un **Índice de Demanda Diaria (IDD)**, que se deriva de nuestro modelo de forecasting de pasajeros. Esto transforma la "Especificación" estática en una lista de empaque dinámica.

La fórmula es:

$$Q_{final} = \lceil \left( \frac{P_{pronosticados}}{\bar{P}_{histórico}} \right) \cdot Q_{base} \rceil$$

**Definición de Variables:**

  * **$Q_{final}$ (Output):** Cantidad final de productos a surtir (entero, redondeado hacia arriba).
  * **$P_{pronosticados}$ (Input):** Número de pasajeros pronosticados por nuestro modelo de IA para el día del vuelo.
  * **$\bar{P}_{histórico}$ (Input/Constante):** Promedio histórico de pasajeros para un día de operación normal.
  * **$Q_{base}$ (Input):** Cantidad de productos estándar (la "Spec Base") que se surte en esa ruta.

### Modelo 3: Estimación de Personal Necesario

Este modelo responde a la pregunta del gerente de planta: "¿Cuántos empleados necesito hoy?". Utiliza el **IDD**, pero basado en el **pronóstico de vuelos** (la principal unidad de trabajo), para escalar la plantilla base.

La fórmula es:

$$E_{necesarios} = \lceil E_{base} \cdot \left( \frac{V_{pronosticados}}{\bar{V}_{histórico}} \right) \rceil$$

**Definición de Variables:**

  * **$E_{necesarios}$ (Output):** Número de empleados recomendados para el turno (entero, redondeado hacia arriba).
  * **$E_{base}$ (Inp   ut/Constante):** El número de empleados necesarios para un día de operación promedio (línea base).
  * **$V_{pronosticados}$ (Input):** Número de vuelos pronosticados por nuestro modelo de IA para ese día.
  * **$\bar{V}_{histórico}$ (Input/Constante):** Promedio histórico de vuelos en un día normal.

-----

## 3\. Especificación de I/O (Contrato de Datos)

Proponemos tres endpoints en nuestra API, uno para cada modelo.

### Endpoint 1: `POST /predict/optimal_batch`

**Descripción:** Obtiene el lote más óptimo de un producto según la lógica FEFO (Modelo 1).

**Input (Request Body):**

```json
{
  "flight_date": "2025-10-28",
  "product_sku": "SKU-12345",
  "available_batches":
}
```

**Output (Response Body):**

```json
{
  "optimal_batch_id": "B1",
  "expiry_date": "2025-11-10"
}
```

### Endpoint 2: `POST /predict/demand`

**Descripción:** Calcula la cantidad ajustada de productos para un vuelo (Modelo 2).

**Input (Request Body):**

```json
{
  "flight_date": "2025-10-28",
  "base_quantity": 450,
  "historical_avg_passengers": 85000
}
```

**Output (Response Body):**

```json
{
  "forecasted_passengers_for_day": 92500,
  "demand_index": 1.088,
  "recommended_quantity": 490
}
```

### Endpoint 3: `POST /predict/staffing`

**Descripción:** Estima el número de empleados necesarios para la operación de un día (Modelo 3).

**Input (Request Body):**

```json
{
  "operation_date": "2025-10-28",
  "baseline_employees": 50,
  "historical_avg_flights": 380
}
```

**Output (Response Body):**

```json
{
  "forecasted_flights_for_day": 425,
  "workload_index": 1.118,
  "recommended_staff": 56
}
```