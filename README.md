# Predicción de Cancelaciones de Reservas Hoteleras
**Máster en Inteligencia Artificial, Cloud Computing y DevOps — PontIA 2026**  
*Módulo de Machine Learning y Deep Learning — Práctica de Evaluación Final*

---

## 👥 Autores y Roles del Equipo

Siguiendo las directrices del guión de entrega, a continuación se detallan las responsabilidades y contribuciones técnicas de cada integrante:

| Integrante | Rol Principal | Contribuciones Clave |
| :--- | :--- | :--- |
| **Daniel Aguilera** | Arquitectura MLOps & Pipeline | Diseño de la arquitectura modular (`data_loader`, `preprocessor`), orquestador principal (`train.py`), módulo de evaluación (`evaluate.py`), modelo de referencia XGBoost con tuning y documentación. |
| **Nitin Babani** | Modelado Machine Learning | Implementación y tuning de modelos basados en árboles y lineales (Árbol de Decisión, Regresión Logística), análisis de métricas e interpretabilidad. |
| **Luis Torres** | Deep Learning & Ensambles | Diseño de la Red Neuronal Multicapa profunda (Keras/TensorFlow), regularización (Dropout, EarlyStopping), Random Forest y tratamiento de alta cardinalidad en `agent`. |

---

## 🎯 Descripción del Problema y Justificación de Negocio

El sector hotelero sufre pérdidas millonarias derivadas de cancelaciones de última hora y *no-shows* (habitaciones vacías que no pueden ser reasignadas a tiempo). 

El objetivo de este proyecto es construir un **sistema automático de Machine Learning** capaz de predecir con alta precisión la probabilidad de que una reserva sea cancelada (`is_canceled = 1`) o confirmada (`is_canceled = 0`), permitiendo al hotel optimizar su política de overbooking y maximizar sus ingresos por habitación disponible (RevPAR).

### Prevención de Data Leakage (Fuga de Datos)
Durante el análisis exploratorio (EDA) se identificaron y eliminaron variables que revelan directamente el desenlace de la reserva:
- `reservation_status`: Registra si el cliente hizo *Check-Out*, *Canceled* o *No-Show*.
- `reservation_status_date`: Fecha en que cambió el estado.

*Ambas fueron excluidas desde la carga inicial (`data_loader.py`) para garantizar un modelo válido y realista en producción.*

---

## ⚙️ Instalación y Configuración del Entorno

1. **Clonar el repositorio**:
   ```bash
   git clone <URL_REPOSITORIO>
   cd practica-machine-learning
   ```

2. **Crear y activar el entorno virtual con `uv` (Python 3.11)**:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Registrar kernel para Jupyter (opcional, para notebooks)**:
   ```bash
   python -m ipykernel install --user --name=practica-ml --display-name="Python (Práctica ML)"
   ```

---

## 🚀 Ejecución del Sistema

El flujo completo está automatizado de principio a fin a través del orquestador central:

### 1. Ejecución Estándar (Modo Rápido / Baseline en ~15s)
Entrena los 5 modelos en su configuración base, evalúa en Test, genera la tabla comparativa y guarda el mejor modelo:
```bash
python -m src.train
```

### 2. Ejecución con Optimización de Hiperparámetros (`--tune`)
Lanza la búsqueda exhaustiva con validación cruzada (`RandomizedSearchCV` / `GridSearchCV` / Keras Tuning) para exprimir el máximo rendimiento:
```bash
python -m src.train --tune
```

### 3. Ejecución individual por modelo
Si se desea depurar o probar un modelo específico de forma aislada:
```bash
python -m src.models.xgboost_model
python -m src.models.neural_network
python -m src.models.random_forest
python -m src.models.decision_tree
python -m src.models.logistic_regresion
```

---

## 📊 Resultados y Comparativa de Modelos

Evaluación de los 5 algoritmos sobre el conjunto de prueba independiente (**Test: 23.878 reservas** estratificadas):

| Modelo | ROC-AUC (Principal) | F1-Score (Secundaria) | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Random Forest (Tuned)** | **0.9609** | **0.8534** | **89.53%** | **88.66%** | 82.26% |
| 🥈 **XGBoost (Tuned)** | 0.9582 | 0.8500 | 88.42% | 81.67% | **88.62%** |
| 🥉 **Red Neuronal (Tuned)** | 0.9537 | 0.8331 | 87.80% | 84.45% | 82.20% |
| 4️⃣ **Decision Tree (Tuned)** | 0.9289 | 0.7883 | 85.16% | 83.57% | 74.60% |
| 5️⃣ **Logistic Regression (Tuned)** | 0.9061 | 0.7730 | 82.16% | 73.10% | 82.02% |

### Justificación de Métricas de Negocio
- **Métrica Principal: ROC-AUC**: Mide la capacidad global del modelo para ordenar las reservas de mayor a menor probabilidad de cancelación en todos los umbrales de decisión. Esto permite al equipo de Revenue Management variar el umbral de corte según la temporada turística (ser más agresivo con el overbooking en temporada alta y más conservador en baja).
- **Métrica Secundaria: F1-Score**: Garantiza un equilibrio saludable entre **Precision** (evitar falsas alarmas que provocarían vender por duplicado una habitación confirmada) y **Recall** (detectar el máximo porcentaje de cancelaciones reales para no dejar habitaciones vacías).

---

## 📈 Visualizaciones y Reportes Generados

Los gráficos se generan automáticamente en la carpeta `reports/figures/`:

1. **Curva ROC Comparativa (`reports/figures/roc_curve_comparison.png`)**:
   Demuestra visualmente cómo la Red Neuronal y XGBoost superan con claridad a los modelos lineales y árboles simples en cualquier umbral operativo.
2. **Matriz de Confusión del Ganador (`reports/figures/confusion_matrix_*.png`)**:
   Permite auditar el recuento de aciertos y el balance de falsos positivos frente a falsos negativos.
3. **Feature Importance (`reports/figures/feature_importance_*.png`)**:
   Identifica los motores de decisión del negocio:
   - `deposit_type_Non Refund`: Factor más determinante (paradoja de cancelaciones masivas de agencias descubierta en el EDA).
   - `market_segment_Online TA`: Gran volatilidad de reservas frente al canal directo.
   - `required_car_parking_spaces`: Factor protector número uno (tasa de cancelación cercana al 0%).
4. **Curvas de Aprendizaje de Keras (`reports/figures/training_history_neural_network.png`)**:
   Evolución de Loss y AUC en entrenamiento vs validación a lo largo de las épocas, mostrando la detención óptima mediante *EarlyStopping*.

---

## 🧪 Registro de Experimentos con MLflow (Bonus Técnico +20%)

El proyecto integra **MLflow Tracking** para garantizar la trazabilidad completa, reproducibilidad y auditoría de todos los experimentos realizados.

### 1. ¿Qué se registra automáticamente en cada ejecución?
- **Experimento**: `Hotel_Cancellations_PontIA`
- **Métricas de Evaluación**: `roc_auc`, `f1_score`, `accuracy`, `precision`, `recall`.
- **Hiperparámetros**: Parámetros de cada algoritmo (tanto configuraciones base como los hiperparámetros óptimos descubiertos en la búsqueda con validación cruzada).
- **Artefactos**:
  - Curva ROC comparativa global (`roc_curve_comparison.png`).
  - Tabla de resumen de métricas (`metrics_summary.csv`).
  - Modelos serializados de producción (`best_model.joblib` y `preprocessor.joblib`).

### 2. Cómo abrir el Dashboard Visual interactivo
Con el entorno virtual activado, ejecuta en la terminal:
```bash
mlflow ui
```
Y abre en tu navegador el enlace directo a los experimentos:
👉 **[http://127.0.0.1:5000/#/experiments/1](http://127.0.0.1:5000/#/experiments/1)**

### 3. Capacidades de auditoría en la UI
- **Pestaña "Runs"**: Tabla dinámica para ordenar y comparar los modelos en tiempo real según cualquier métrica.
- **Herramienta "Compare"**: Selección múltiple de modelos (por ejemplo, *XGBoost Base* vs *XGBoost Tuned*) para generar automáticamente gráficos de barras comparativos y matrices de diferencias de hiperparámetros.
- **Pestaña "Artifacts"**: En la corrida `🏆 Modelo Ganador y Comparativa`, permite inspeccionar la Curva ROC comparativa incrustada directamente en la web y descargar los artefactos de inferencia con un solo clic.

---

## 🔮 Inferencia y Recomendaciones de Negocio en Producción (`src/predict.py`)

El sistema no se detiene en la evaluación académica: cuenta con un módulo de inferencia en tiempo real ([`src/predict.py`](src/predict.py)) que consume directamente los artefactos persistidos en `models/` (`preprocessor.joblib` y `best_model.joblib`).

### 1. Ejecutar la Simulación de Predicción
```bash
python -m src.predict
```

### 2. Soporte Dual: Tiempo Real (Online) y Lotes (Batch)
La función `predict_reservation(data)` implementa una arquitectura flexible:
- **Modo Online (`dict`)**: Permite alimentar peticiones HTTP de una API REST (FastAPI) o formularios web (Streamlit) pasando un diccionario con los datos de **1 sola reserva**.
- **Modo Batch (`DataFrame`)**: Permite procesar de golpe un archivo CSV con **cientos de reservas** para generar informes masivos de ocupación y riesgo en milisegundos.

### 3. Matriz de Decisión Operativa (Revenue Management)
El modelo no devuelve un frío `0` o `1`, sino que traduce la probabilidad matemática en **niveles de riesgo accionables** basados en la tasa histórica del hotel (37%):

| Nivel de Riesgo | Rango de Probabilidad | Diagnóstico de Negocio | Acción Operativa Recomendada |
| :---: | :---: | :--- | :--- |
| 🟢 **BAJO** | $< 40\%$ | Por debajo de la media histórica (37%). Clientes con parking, reservas directas o con peticiones especiales. | **No intervenir**: Mantener habitación asignada y evitar comunicaciones innecesarias para cuidar la experiencia. |
| 🟡 **MEDIO** | $40\% - 69\%$ | Incertidumbre moderada. Probabilidad por encima de la media histórica. | **Acción preventiva suave**: Envío de email de cortesía 48h antes reconfirmando hora estimada de llegada. |
| 🔴 **ALTO** | $\ge 70\%$ | Alta probabilidad de habitación vacía. Típico en reservas *Non Refund* de OTAs con alta antelación. | **Protección de ingresos**: Activar overbooking preventivo y solicitar preautorización/tarjeta de crédito de garantía. |

---

## 🖥️ Aplicación Web Interactiva (`app.py`)

Para acercar el modelo a los usuarios finales (recepcionistas y analistas de *Revenue Management*), el proyecto incluye una aplicación web completa desarrollada con **Streamlit** ([`app.py`](app.py)), desacoplada del pipeline de entrenamiento y conectada directamente con el motor de inferencia.

### 1. Cómo Ejecutar la Aplicación
Con el entorno virtual activado, ejecuta en la terminal:
```bash
streamlit run app.py
```
Acceso en el navegador: 👉 **[http://localhost:8501](http://localhost:8501)**

### 2. Estructura y Módulos de la Aplicación

La interfaz está dividida en **3 pestañas de negocio** y una **barra lateral informativa**:

#### 🛎️ Pestaña 1: Predicción Individual (Front-Desk)
- **Formulario temático en 3 columnas**: Organizado de forma natural para el flujo de trabajo en recepción (*Hotel y Calendario*, *Huéspedes y Servicios*, *Garantía y Canal*).
- **Semáforo de Riesgo en Tiempo Real**: Devuelve de forma instantánea la probabilidad estimada y categoriza el riesgo con alertas visuales de color (🟢 **Bajo**, 🟡 **Medio**, 🔴 **Alto**).
- **Prescripción Operativa**: Traduce la predicción matemática en acciones prácticas concretas (solicitar tarjeta de garantía, enviar recordatorio o mantener habitación sin molestar al huésped).

#### 📁 Pestaña 2: Auditoría por Lotes (Revenue Management)
- **Carga de Archivos CSV (`st.file_uploader`)**: Permite arrastrar un archivo con cientos de reservas futuras para auditoría masiva.
- **Botón de Prueba Rápida en 1 Clic**: Incluye un dataset de muestra preconfigurado (`data/sample_batch.csv` con 30 reservas) para realizar demostraciones en vivo sin necesidad de archivos externos.
- **Resumen Ejecutivo con KPIs**: Tarjetas métricas automáticas (*Total Reservas*, *Tasa de Cancelación Prevista*, *Volumen en Riesgo Alto y Bajo*).
- **Gráfico de Distribución**: Visualización rápida de barras con el balance de riesgo del lote.
- **Exportación de Informes (`st.download_button`)**: Permite descargar un CSV enriquecido con la probabilidad, el nivel de riesgo y la recomendación asignada a cada reserva.

#### 📊 Pestaña 3: Benchmark y Evaluación Técnica
- **Tabla Resumen Dinámica**: Muestra las métricas de test de los 5 modelos con resaltado automático del mejor valor (diseñado con alto contraste compatible tanto en *Dark Mode* como en *Light Mode*).
- **Curva ROC Multi-Modelo**: Gráfica comparativa superpuesta generada por el orquestador.
- **Detalles Colapsables (`st.expander`)**: Permite al evaluador inspeccionar la *Feature Importance* (variables más determinantes) y la *Matriz de Confusión* del modelo ganador.

#### 📌 Barra Lateral (Sidebar)
- **Identidad del Proyecto**: Autores (Daniel Aguilera, Luis Torres, Nitin Babani) y tutor (Sergio Benito).
- **Ficha Técnica en Vivo**: Muestra el nombre y métricas del modelo ganador actualmente en producción.
- **Acceso Directo a MLOps**: Botón de enlace nativo (`st.link_button`) para abrir el panel de **MLflow** en una nueva pestaña (`http://localhost:5000`).
- **Chuleta Operativa**: Resumen rápido de los umbrales de decisión del hotel.

### 3. Decisiones de Arquitectura y Rendimiento (MLOps)
- **Caché en Memoria RAM (`@st.cache_resource`)**: Dado que el modelo Random Forest serializado pesa más de 400 MB, la aplicación utiliza `@st.cache_resource` para cargarlo en memoria una única vez al arrancar, evitando lecturas reiteradas de disco y garantizando respuestas en milisegundos en cada interacción.
- **Cero Valores a Fuego (*Zero-Hardcoding*)**: El dashboard lee dinámicamente `reports/metrics_summary.csv` (`@st.cache_data`) para detectar de forma automática qué algoritmo ganó el benchmark y reflejar sus métricas reales en el sidebar y en las pestañas, desacoplando completamente la interfaz de cualquier modelo concreto.

---

## 📁 Estructura del Proyecto

```text
practica-machine-learning/
├── app.py                               # Dashboard web interactivo en producción (Streamlit)
├── data/
│   ├── dataset_practica_final.csv       # Dataset original (119.390 reservas)
│   └── sample_batch.csv                 # Muestra de 30 reservas para auditoría por lotes
├── models/                              # Artefactos persistidos para inferencia
│   ├── best_model.joblib                # Modelo ganador seleccionado automáticamente
│   └── preprocessor.joblib              # ColumnTransformer ajustado para inferencia
├── notebooks/
│   └── 01_eda.ipynb                     # Análisis exploratorio detallado (EDA)
├── reports/
│   ├── metrics_summary.csv              # Tabla resumen de métricas
│   └── figures/                         # Gráficas generadas automáticamente
│       ├── roc_curve_comparison.png     # Curva ROC comparativa entre modelos
│       ├── training_history_*.png       # Curvas de convergencia de Keras
│       ├── confusion_matrix_*.png       # Matrices de confusión
│       └── feature_importance_*.png     # Importancia relativa de variables
├── src/
│   ├── config.py                        # Rutas globales, semillas y listas de columnas
│   ├── data_loader.py                   # Carga de datos, filtrado y división 80/20
│   ├── preprocessor.py                  # Imputación, StandardScaler y OneHotEncoder
│   ├── evaluate.py                      # Métricas de negocio y funciones de visualización
│   ├── train.py                         # Orquestador maestro del pipeline
│   ├── predict.py                       # Inferencia unitaria y por lotes con reglas de negocio
│   └── models/                          # Módulos desacoplados de modelado
│       ├── __init__.py
│       ├── xgboost_model.py             # Gradient Boosting (XGBoost)
│       ├── neural_network.py            # Red Neuronal Profunda (TensorFlow / Keras)
│       ├── random_forest.py             # Random Forest Classifier
│       ├── decision_tree.py             # Árbol de Decisión
│       └── logistic_regresion.py        # Regresión Logística
├── README.md                            # Documentación principal
└── requirements.txt                     # Dependencias del proyecto (incluye mlflow y streamlit)
```
