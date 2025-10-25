# Resumen de Validación del Problema: gategroup

Este documento resume los hallazgos de nuestro Análisis Exploratorio de Datos (EDA) sobre el dataset proporcionado por gategroup.

## 1. El Problema

Tras analizar los datos de operaciones de las plantas 1 y 2, nuestro equipo observó un problema clave: la alta volatilidad diaria en la carga de trabajo y la falta de granularidad en los datos disponibles.    

Los datos muestran fluctuaciones significativas y patrones estacionales en el número de vuelos y pasajeros diarios. Por ejemplo, la carga de trabajo puede variar drásticamente de un día para otro, especialmente en torno a festividades o fines de semana. Esta volatilidad, combinada con el hecho de que los datos están agregados a nivel diario, hace que la planificación táctica sea extremadamente difícil.

Este problema a nivel de datos es la causa raíz de los dos desafíos principales mencionados en la presentación del reto :   

Productividad Inconsistente (Productivity Estimation): La incapacidad de prever los picos de trabajo conduce a una asignación de personal reactiva, generando cuellos de botella y una gran variabilidad en los tiempos de empaque (de 3.5 a 7 minutos por el mismo carrito).    

(El siguiente gráfico, generado a partir de Plant 1.csv, ilustra la volatilidad diaria de pasajeros, un indicador clave de la carga de trabajo impredecible que enfrenta el personal.)

* **Desperdicio Masivo (Consumption Prediction):** Sin una forma de anticipar la demanda diaria, la operación recurre a un modelo de "rellenado" estático, lo que provoca que los vuelos regresen con más del 50% de los artículos sin usar.  
* **Productividad Inconsistente (Productivity Estimation)** La incapacidad de prever los picos de trabajo conduce a una asignación de personal reactiva, generando cuellos de botella y una gran variabilidad en los tiempos de empaque (de 3.5 a 7 minutos por el mismo carrito).  

(grafico)

## 2. La Oportunidad
Los datos sugieren que, aunque la planificación actual es reactiva, la carga de trabajo no es completamente aleatoria. Nuestra hipótesis es que la volatilidad en los vuelos y pasajeros contiene patrones temporales (semanales, mensuales, estacionales) que son predecibles con un alto grado de precisión.

La oportunidad no reside en obtener datos más granulares (que no tenemos), sino en transformar los datos operativos de alto nivel (macro) en una herramienta de inteligencia táctica (micro). Creemos que podemos usar un pronóstico preciso de la carga de trabajo diaria para modular las operaciones a nivel de carrito y de personal, creando un puente entre la planificación estratégica y la ejecución en la planta.

## 3. Nuestra Solución Propuesta

Basado en este análisis, nuestro proyecto se centrará en desarrollar el Sistema de Planificación Inteligente de Recursos (SPIR). Esta solución ataca directamente los problemas de desperdicio y productividad utilizando únicamente los datos proporcionados.

Nuestra solución consiste en:

* **Construir un Modelo de Forecasting de Alta Precisión:** Utilizando un algoritmo XGBoost, entrenaremos un modelo con los datos de series temporales (Plant 1.csv, Plant 2.csv) para predecir la carga de trabajo (vuelos y pasajeros) de los próximos días con un error inferior al 2%, cumpliendo con el objetivo del reto
* **Crear un "Índice de Demanda Diaria" (IDD):** Traduciremos el pronóstico del modelo en un índice simple que sirva como guía para la operación diaria:

Para la Predicción de Consumo: El IDD de Pasajeros ajustará dinámicamente la cantidad de productos a cargar en los carritos, reduciendo el desperdicio en días de baja demanda y evitando la escasez en días de alta demanda.

Para la Estimación de Productividad: El IDD de Vuelos proporcionará a los gerentes una recomendación clara sobre el número de empleados necesarios para el día, permitiendo una planificación proactiva que estabilice la productividad y evite los cuellos de botella.