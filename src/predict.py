"""
Predicción cargando modelo pre-entrenado
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src import config
from src.preprocessor import clean_features

_PREPROCESSOR = None
_MODEL = None


def load_artifacts():
    """Carga en memoria el preprocesador y el mejor modelo entrenado."""
    global _PREPROCESSOR, _MODEL

    prep_path = config.MODELS_DIR / "preprocessor.joblib"
    model_path = config.MODELS_DIR / "best_model.joblib"

    if not prep_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"No se encontraron los artefactos necesarios en {config.MODELS_DIR}. "
            "Por favor ejecuta primero 'python -m src.train' para generarlos."
        )

    if _PREPROCESSOR is None:
        _PREPROCESSOR = joblib.load(prep_path)

    if _MODEL is None:
        _MODEL = joblib.load(model_path)

    return _PREPROCESSOR, _MODEL


def predict_reservation(data: dict | pd.DataFrame) -> dict | list[dict]:
    """
    Predice si una o varias reservas serán canceladas y estima su probabilidad.
    """
    preprocessor, model = load_artifacts()

    # Convertimos a DataFrame si es un diccionario
    if isinstance(data, dict):
        df_input = pd.DataFrame([data])
        is_single = True
    else:
        df_input = data.copy()
        is_single = False

    # 1. Aplicamos la limpieza de features
    df_clean = clean_features(df_input)

    # 2. Transformamos los datos con el preprocesador
    X_prep = preprocessor.transform(df_clean)

    # 3. Predicción de probabilidades y clases
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_prep)[:, 1]
        preds = model.predict(X_prep)
    else:
        # Compatibilidad con Keras / Redes Neuronales
        X_array = np.asarray(X_prep, dtype=np.float32)
        probas = model.predict(X_array, verbose=0).ravel()
        preds = (probas >= 0.5).astype(int)

    results = []
    for proba, pred in zip(probas, preds):
        if proba >= 0.70:
            riesgo = "ALTO"
            recomendacion = "Alerta: Aplicar overbooking preventivo y solicitar prepago/tarjeta de garantía."
        elif proba >= 0.40:
            riesgo = "MEDIO"
            recomendacion = "Reconfirmar asistencia enviando email de cortesía 48h antes."
        else:
            riesgo = "BAJO"
            recomendacion = "Reserva muy segura. Mantener habitación asignada."

        results.append({
            "is_canceled": int(pred),
            "cancellation_probability": round(float(proba), 4),
            "risk_level": riesgo,
            "recommendation": recomendacion
        })

    return results[0] if is_single else results


if __name__ == "__main__":
    print("=" * 65)
    print("🏨 SIMULADOR DE INFERENCIA EN PRODUCCIÓN (HOTEL CANCELLATIONS)")
    print("=" * 65)

    sample_low_risk = {
        "hotel": "Resort Hotel",
        "lead_time": 14,
        "arrival_date_year": 2017,
        "arrival_date_month": "August",
        "arrival_date_week_number": 33,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 1,
        "babies": 0,
        "meal": "BB",
        "country": "ESP",
        "market_segment": "Direct",
        "distribution_channel": "Direct",
        "is_repeated_guest": 1,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 2,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 1,
        "deposit_type": "No Deposit",
        "agent": 0.0,
        "company": None,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 120.0,
        "required_car_parking_spaces": 1,
        "total_of_special_requests": 2,
    }

    sample_high_risk = {
        "hotel": "City Hotel",
        "lead_time": 280,
        "arrival_date_year": 2017,
        "arrival_date_month": "May",
        "arrival_date_week_number": 20,
        "arrival_date_day_of_month": 18,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 4,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 2,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "Non Refund",
        "agent": 9.0,
        "company": None,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 98.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
    }

    res_low = predict_reservation(sample_low_risk)
    res_high = predict_reservation(sample_high_risk)

    print("\n🟢 CASO 1: Huésped Directo / Fidelizado / Con Parking")
    print(f"  • Probabilidad de Cancelación : {res_low['cancellation_probability'] * 100:.1f}%")
    print(f"  • Nivel de Riesgo             : {res_low['risk_level']}")
    print(f"  • Acción Operativa Sugerida   : {res_low['recommendation']}")

    print("\n🔴 CASO 2: Agencia Externa / Non Refund / Alta Antelación")
    print(f"  • Probabilidad de Cancelación : {res_high['cancellation_probability'] * 100:.1f}%")
    print(f"  • Nivel de Riesgo             : {res_high['risk_level']}")
    print(f"  • Acción Operativa Sugerida   : {res_high['recommendation']}")
    print("\n" + "=" * 65)