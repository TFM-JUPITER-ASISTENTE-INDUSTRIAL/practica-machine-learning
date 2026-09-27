import streamlit as st
import pandas as pd
from src.predict import load_artifacts, predict_reservation

st.set_page_config(
    page_title="Predicción de Cancelaciones | Hotel AI",
    layout="wide",
)

# Carga del modelo en memoria caché
@st.cache_resource
def get_model():
    return load_artifacts()

with st.spinner("Cargando modelo en memoria..."):
    preprocessor, model = get_model()
st.toast("Modelo listo", icon="✅")


@st.cache_data
def load_metrics_data():
    """Lee el resumen de métricas y detecta dinámicamente el modelo ganador."""
    try:
        df = pd.read_csv("reports/metrics_summary.csv")
        # El ganador es el de mayor ROC-AUC
        best_row = df.sort_values(by="roc_auc", ascending=False).iloc[0]
        return df, best_row
    except Exception:
        return None, None

# Cargamos métricas
df_metrics, best_model_info = load_metrics_data()

with st.sidebar:
    st.header("Hotel Predicciones")
    st.markdown("**PontIA - Módulo Machine Learning**")
    st.caption("Tutor: Sergio Benito")

    st.markdown("---")
    st.subheader("Equipo de Desarrollo")
    st.markdown("""
    - **Daniel Aguilera**: MLOps, Preprocesador y XGBoost
    - **Luis Torres**: Regresión Logística y Árbol de Decisión
    - **Nitin Babani**: Red Neuronal y Random Forest
    """)

    st.markdown("---")
    st.subheader("Modelo Ganador")
    if best_model_info is not None:
        winner_name = best_model_info["model"]
        winner_roc = f"{best_model_info['roc_auc']:.4f}"
        winner_f1 = f"{best_model_info['f1_score']:.4f}"
    else:
        # Fallback dinámico si no hay CSV: leemos el nombre de la clase del modelo
        winner_name = type(model).__name__
        winner_roc = "N/A"
        winner_f1 = "N/A"

    st.info(f"**{winner_name}**")

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("ROC-AUC", winner_roc)
    with col_sb2:
        st.metric("F1-Score", winner_f1)

    st.markdown("---")
    st.subheader("MLOps Tracking")
    st.link_button("Abrir MLflow Dashboard", "http://localhost:5000", use_container_width=True)

    st.markdown("---")
    st.subheader("Matriz de Decisión")
    st.markdown("""
    - 🟢 **Bajo (< 40%)**: Confirmar sin fricción.
    - 🟡 **Medio (40-69%)**: Email cortesía 48h antes.
    - 🔴 **Alto (≥ 70%)**: Overbooking y fianza.
    """)

tab1, tab2, tab3 = st.tabs([
    "Predicción Individual",
    "Predicción por lotes",
    "Métricas"
])

# Formulario de reserva
with tab1:
    st.markdown("### DATOS DE LA RESERVA")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Hotel y Calendario")
        hotel = st.selectbox("Tipo de Hotel", ["City Hotel", "Resort Hotel"])
        lead_time = st.slider("Dias de antelación (Lead Time)", min_value=0, max_value=500, value=10)
        # TODO Selector de rango de fechas y autocalculo de valores siguientes
        arrival_month = st.selectbox("Mes de llegada",
                                     ["January", "February", "March", "April",
                                      "May", "June", "July","August", "September",
                                      "October","November", "December"], index=6)
        stays_week = st.number_input("Noches entre semana", min_value=0, max_value=20, value=2)
        stays_weekend = st.number_input("Noches fin de semana", min_value=0, max_value=10, value=1)
        adr = st.number_input("Precio medio por noche (ADR)", min_value=0.0, max_value=1000.0, value=95.0, step=5.0)

    with col2:
        st.subheader("Huéspedes y Servicios")
        adults = st.number_input("Adultos", min_value=1, max_value=10,
                                 value=2)
        children = st.number_input("Niños", min_value=0, max_value=10,
                                   value=0)
        babies = st.number_input("Bebés", min_value=0, max_value=10,
                                 value=0)
        meal = st.selectbox("Régimen de comidas", ["BB", "HB", "FB",
                                                   "SC"])
        room_type = st.selectbox("Tipo de habitación reservada", ["A", "B",
                                                                  "C", "D", "E", "F", "G"])
        parking = st.selectbox("¿Requiere plaza de parking?", [0, 1],
                               format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
        special_requests = st.slider("Peticiones especiales", min_value=0,
                                     max_value=5, value=1)

    with col3:
        st.subheader("Garantía y Canal")
        deposit_type = st.selectbox("Tipo de Depósito", ["No Deposit","Non Refund", "Refundable"])
        market_segment = st.selectbox("Segmento de Mercado", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"])
        distribution_channel = st.selectbox("Canal de Distribución", ["TA/TO", "Direct", "Corporate", "GDS"])
        customer_type = st.selectbox("Tipo de Cliente", ["Transient", "Transient-Party", "Contract", "Group"])
        prev_cancellations = st.number_input("Cancelaciones previas del cliente", min_value=0, max_value=20, value=0)
        country = st.selectbox("País de origen", ["PRT", "GBR", "FRA", "ESP", "DEU", "Other"], index=3)
        agent_id = st.selectbox(
            "Agencia de viajes (Agent)",
            [0.0, 9.0, 240.0, 1.0, 14.0, 7.0, 6.0, 250.0, "Other"],
            format_func=lambda x: "Directo / Sin Agencia (0)" if x == 0.0
            else f"Agencia #{x}"
        )

    # EVALUACIÓN
    st.markdown("---")

    if st.button("Evaluar Riesgo de Cancelación", type="primary", use_container_width=True):
        reservation_data = {
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": 2027,
            "arrival_date_month": arrival_month,
            "arrival_date_week_number": 28,
            "arrival_date_day_of_month": 15,
            "stays_in_weekend_nights": stays_weekend,
            "stays_in_week_nights": stays_week,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": 0,
            "previous_cancellations": prev_cancellations,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": room_type,
            "assigned_room_type": room_type,
            "booking_changes": 0,
            "deposit_type": deposit_type,
            "agent": agent_id,
            "company": None,
            "days_in_waiting_list": 0,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": parking,
            "total_of_special_requests": special_requests,
        }

        result = predict_reservation(reservation_data)

        proba = result["cancellation_probability"] * 100
        riesgo = result["risk_level"]
        recomendacion = result["recommendation"]

        st.markdown("### Diagnóstico de Revenue Management")

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(label="Probabilidad Estimada de Cancelación",
                      value=f"{proba:.1f} %")
        with col_metric2:
            st.metric(label="Nivel de Riesgo Operativo", value=riesgo)

        if riesgo == "ALTO": st.error(f"**RIESGO ALTO ({proba:.1f}%)**: {recomendacion}")
        elif riesgo == "MEDIO":
            st.warning(f"**RIESGO MEDIO ({proba:.1f}%)**: {recomendacion}")
        else:
            st.success(f"**RIESGO BAJO ({proba:.1f}%)**: {recomendacion}")

with tab2:
    st.subheader("Reservas por Lotes")
    st.markdown(
        "Sube un archivo CSV con las reservas a auditar para calcular probabilidades de cancelación."
    )

    # Dos opciones: Subir archivo o usar el dataset de prueba
    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("Cargar Dataset de Prueba (30 reservas)")

    df_batch = None
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.toast(f"Archivo cargado: **{uploaded_file.name}** ({len(df_batch)} reservas)")
    elif use_sample:
        df_batch = pd.read_csv("data/sample_batch.csv")
        st.info(f"Dataset de prueba cargado ({len(df_batch)} reservas)")

    # Si hay datos cargados, ejecutamos la prediccion
    if df_batch is not None:
        with st.spinner("Analizando reservas con el modelo..."):
            # Llamamos a nuestra función pasando el DataFrame completo
            results = predict_reservation(df_batch)
            df_results = pd.DataFrame(results)

            # Unimos los resultados al DataFrame original
            df_final = pd.concat([df_batch, df_results], axis=1)

        st.markdown("---")
        st.markdown("### Resumen Ejecutivo del Lote")

        # 2. Métricas globales
        total_res = len(df_final)
        pct_cancel = (df_final["is_canceled"].sum() / total_res) * 100
        n_alto = (df_final["risk_level"] == "ALTO").sum()
        n_medio = (df_final["risk_level"] == "MEDIO").sum()
        n_bajo = (df_final["risk_level"] == "BAJO").sum()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Reservas", f"{total_res}")
        kpi2.metric("Cancelación Prevista", f"{pct_cancel:.1f} %")
        kpi3.metric("🔴 Riesgo Alto", f"{n_alto}")
        kpi4.metric("🟢 Riesgo Bajo", f"{n_bajo}")

        # 3. Gráfico rápido de distribución del riesgo
        st.markdown("#### Distribución de Niveles de Riesgo")
        risk_counts = df_final["risk_level"].value_counts()
        st.bar_chart(risk_counts)

        # 4. Tabla interactiva con los resultados
        st.markdown("#### Detalle de Reservas y Acciones")
        # Mostramos primero las columnas clave de negocio
        cols_to_show = ["risk_level", "cancellation_probability", "recommendation",
                        "hotel", "lead_time", "adr", "deposit_type", "market_segment"]
        existing_cols = [c for c in cols_to_show if c in df_final.columns]
        st.dataframe(df_final[existing_cols], use_container_width=True)

        # 5. Botón para exportar los resultados a CSV
        csv_bytes = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar Informe Completo en CSV",
            data=csv_bytes,
            file_name="informe_riesgo_cancelaciones.csv",
            mime="text/csv",
            use_container_width=True
        )

with tab3:
    st.subheader("Comparativa de Modelos")
    st.markdown(
        "Resultados sobre el conjunto de test (20%, 23.878 reservas) "
    )

    # 1. Tabla de Métricas
    st.markdown("#### Tabla Resumen")
    if df_metrics is not None:
        st.dataframe(
            df_metrics.style.highlight_max(
                axis=0,
                subset=["roc_auc", "f1_score", "accuracy", "recall"],
                props="background-color: #1b4332; color: #d8f3dc; font-weight: bold;"
            ),
            use_container_width=True
        )

    st.markdown("---")

    # 2. La Curva ROC Comparativa
    col_roc, col_info = st.columns([3, 2])

    with col_roc:
        st.markdown("#### Curva ROC Multi-Modelo")
        try:
            st.image(
                "reports/figures/roc_curve_comparison.png",
                caption="Curva ROC comparativa generada por src/evaluate.py",
                use_container_width=True
            )
        except Exception:
            st.info("Imagen reports/figures/roc_curve_comparison.png no encontrada.")

    with col_info:
        st.markdown(f"#### ¿Por qué ganó {winner_name}?")
        st.markdown(f"""
        - **ROC-AUC ({winner_roc})**.
        - **F1-Score ({winner_f1})**.
        """)

    # 3. Desplegables Detalles
    with st.expander("Ver Importancia de Variables"):
        st.markdown("Variables con mayor peso predictivo en la decisión del modelo:")
        try:
            st.image(
                "reports/figures/feature_importance_random_forest_(tuned).png",
                caption="Top variables más influyentes según Random Forest",
                use_container_width=True
            )
        except Exception:
            st.info("Gráfico de feature importance no disponible.")

    with st.expander("Ver Matriz de Confusión"):
        st.markdown("Distribución de aciertos y errores en el conjunto de test:")
        try:
            st.image(
                "reports/figures/confusion_matrix_random_forest_(tuned).png",
                caption="Matriz de confusión del modelo ganador",
                use_container_width=True
            )
        except Exception:
            st.info("Matriz de confusión no disponible.")

