import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import lineStyles
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from src import config

FIGURES_DIR = config.REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true, y_pred, y_proba=None):
    """ Calcula métricas de evaluación
        Métricas:
        - Accuracy : Porcentaje global de aciertos TP+TN/Total
        - Precision : Fiabilidad de lo que predijo como cancelación que % acertó
        - Recall : Cobertura de cancelaciones: de todas las que cancelaron que % detectó
        - f1_score : Media armónica entre Precision y Recall (mide el equilibrio entre ambas)
        - roc_auc : Capacidad del modelo para discriminar entre clases en cualquier umbral

        Importancia de cada métrica:
        - La precision mide de las predicciones de cancelación cuantas fueron reales y cuantas
        fueron falsas alarmas, es muy importante, si el modelo tiene alta precision permite al hotel
        aplicar overbooking para mitigar las cancelaciones previstas.
        - Recall mide de todas las cancelaciones que ocurrieron de verdad cuantas fue capaz de
        predecir el modelo. Es igual de importante que la precision para este caso, si fuera bajo
        podrian quedarse habitaciones vacias
        - F1 Score: Como para el hotel es tan grave vender una habitación de mas como dejar una habitación
        vacia, esta metríca es mas importante que las anteriores porque mide que el modelo sea
        equilibrado.
        - ROC-AUC mide la calidad del modelo, sin importar el umbral que se elija para la predicción
        el hotel podría decidir tomar decisiones en base al umbral de prediccion usando model.predict_proba()
        para dependiendo de la epoca del año adoptar mejores medidas en base a las predicciones

        Métrica Principal para evaluar el modelo: ROC-AUC
        Métrica Secundaria F1 Score
    """
    metrics = {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall" : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score" : round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    else:
        metrics["roc_auc"] = None
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name:str = "modelo", save_path=None):
    """ Genera, muestra y guarda la Matriz de confusión """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No cancela (0)" , "Cancela (1)"]

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(f"Matriz de Confusión - {model_name}", fontsize=13, pad=12)
    plt.xlabel("Predicción del Modelo", fontsize=11)
    plt.ylabel("Realidad", fontsize=11)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"

    plt.savefig(save_path, dpi=150)
    print(f"Matriz de confusión guardada en: {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name:str = "modelo", save_path=None):
    """ Genera, muestra y guarrda la curva ROC """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"{model_name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label="Azar (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (Falsas Alarmas)", fontsize=11)
    plt.ylabel("Tasa de Verdaderos Positivos (Recall / Cazados", fontsize=11)
    plt.title(f"Curva ROC - {model_name}", fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"

    plt.savefig(save_path, dpi=150)
    print(f"Curva ROC guardada en: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, top_n: int = 15, model_name:str = "modelo", save_path=None):
    """ Genera y guarda un gráfico de barras con las variables más importantes del modelo """
    if not hasattr(model, "feature_importances_"):
        print(f"El modelo {model_name} no tiene el atributo feature_importance_")
        return

    # Limpiamos prefijos para que los nombres sean legibles
    clean_names = [
        f.replace("num__", "").replace("cat__", "")
        for f in feature_names
    ]

    # Dataframe ordenados de mayor a menor y cojemos el Top N
    df_imporance = pd.DataFrame({
        "feature" : clean_names,
        "importance" : model.feature_importances_,
    }).sort_values(by="importance", ascending=True).tail(top_n)

    # Dibujamos el gráfico
    plt.figure(figsize=(9,6))
    plt.barh(df_imporance["feature"], df_imporance["importance"], color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.xlabel("Importancia Relativa", fontsize=11)
    plt.ylabel("Característica (Feature)", fontsize=11)
    plt.title(f"Top {top_n} Variables mas importantes - {model_name}", fontsize=13, pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"

    plt.savefig(save_path, dpi=150)
    print(f"Feature importance guardada en: {save_path}")
    plt.close()

if __name__ == "__main__":
    from src.data_loader import get_train_test_data
    from src.preprocessor import preprocess_data
    from src.models import xgboost_model

    print("Cargando datos y entrenando XGBoost para probar la matriz...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    X_train_prep, X_test_prep, preprocessor = preprocess_data(X_train, X_test)

    model = xgboost_model.get_model()
    model.fit(X_train_prep, y_train)
    y_pred = model.predict(X_test_prep)
    y_proba = model.predict_proba(X_test_prep)[:, 1]

    # Probamos las dos funciones
    metricas = compute_metrics(y_test, y_pred, y_proba)
    print("Métricas calculadas:", metricas)

    plot_confusion_matrix(y_test, y_pred, model_name="XGBoost")

    plot_roc_curve(y_test, y_proba, model_name="XGBoost")

    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(model, feature_names, top_n=15, model_name="XGBoost")