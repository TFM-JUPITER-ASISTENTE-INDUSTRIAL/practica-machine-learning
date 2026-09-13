# Máster PONTIA 2026 - Predicción de Cancelaciones Hoteleras

Proyecto de Machine Learning para predecir si una reserva de hotel será cancelada (`is_canceled = 1` o `0`).

## Autores
- Luis Torres
- Nitin Babani
- Daniel Aguilera

## Instalación y entorno
1. Crear el entorno virtual:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   ```
2. Instalar dependencias:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Registrar el kernel en Jupyter:
   ```bash
   python -m ipykernel install --user --name=practica-ml --display-name="Python (Práctica ML)"
   ```

## Estructura del proyecto
```text
practica-machine-learning/
├── data/
│   └── dataset_practica_final.csv       # Dataset original
├── notebooks/
│   └── 01_eda.ipynb                     # Análisis exploratorio (EDA)
├── src/
│   ├── config.py                        # Rutas, constantes y definición de columnas
│   ├── data_loader.py                   # Carga de datos, limpieza de outliers y split train/test
│   ├── preprocessor.py                  # Limpieza y ColumnTransformer (StandardScaler + OneHotEncoder)
│   └── models/                          # Modelos
│       ├── __init__.py
│       └── xgboost_model.py             # Modelo XGBoost (referencia con tuning)
├── README.md
└── requirements.txt
```

## Ejecución
Para probar el modelo XGBoost implementado:
```bash
uv run python -m src.models.xgboost_model
```
