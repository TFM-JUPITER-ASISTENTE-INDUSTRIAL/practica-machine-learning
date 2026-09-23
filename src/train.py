"""
Script principal de Entrenamiento y Orquestación

1. Carga Datos
2. Prepocesamiento
3. Carga de modelos
4. Entreamiento y Evaluación
5. Tabla comparativa
6. Selección y guardado del mejor modelo
7. Graficos
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from src import config
from src.data_loader import get_train_test_data
from src.preprocessor import preprocess_data
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

def load_available_models(tune: bool = False, X_train = None, y_train = None):
    models = {}

    # 1. XGBoost
    try:
        from src.models import xgboost_model
        if tune and hasattr(xgboost_model, "tune_model") and X_train is not None:
            print("\n[INFO] Optimizando XGBoost...")
            best_model, _ = xgboost_model.tune_model(X_train, y_train)
            models["XGBoost (Tuned)"] = best_model
        else:
            models["XGBoost"] = xgboost_model.get_model()
    except (ModuleNotFoundError, ImportError):
        pass
    except Exception as e:
        print(f"[WARINIG] Eror al cargar XGBoost: {e}")

    # 2. Regresión logística
    try:
        from src.models import logistic_regression
        if tune and hasattr(logistic_regression, "tune_model") and X_train is not None:
            print("\n[INFO] Optimizando Regresión Logística...")
            best_model, _ = logistic_regression.tune_model(X_train, y_train)
            models["Logistic Regression (Tuned)"] = best_model
        else:
            models["Logistic Regression"] = logistic_regression.get_model()
    except (ModuleNotFoundError, ImportError):
        pass
    except Exception as e:
        print(f"[WARINIG] Eror al cargar Regresión Logística: {e}")

    # 3. Árbol de Decisión
    try:
        from src.models import decision_tree
        if tune and hasattr(decision_tree, "tune_model") and X_train is not None:
            print("\n[INFO] Optimizando Decision Tree...")
            best_model, _ = decision_tree.tune_model(X_train, y_train)
            models["Decision Tree (Tuned)"] = best_model
        else:
            models["Decision Tree"] = decision_tree.get_model()
    except (ModuleNotFoundError, ImportError):
        pass
    except Exception as e:
        print(f"[WARINIG] Eror al cargar Decision Tree: {e}")

    # 4 Random Forest
    try:
        from src.models import random_forest
        if tune and hasattr(random_forest, "tune_model") and X_train is not None:
            print("\n[INFO] Optimizando Random Forest...")
            best_model, _ = random_forest.tune_model(X_train, y_train)
            models["Random Forest (Tuned)"] = best_model
        else:
            models["Random Forest"] = random_forest.get_model()
    except (ModuleNotFoundError, ImportError):
        pass
    except Exception as e:
        print(f"[WARINIG] Eror al cargar Random Forest: {e}")

    # 5 Red Neuronal
    try:
        from src.models import neural_network
        if tune and hasattr(neural_network, "tune_model") and X_train is not None:
            print("\n[INFO] Optimizando Red Neuronal...")
            best_model, _ = neural_network.tune_model(X_train, y_train)
            models["Red Neuronal (Tuned)"] = best_model
        else:
            models["Red Neuronal"] = neural_network.get_model(input_dim=X_train.shape[1])
    except (ModuleNotFoundError, ImportError):
        pass
    except Exception as e:
        print(f"[WARINIG] Eror al cargar Red Neuronal: {e}")

    return models

def run_pipeline(tune: bool = False):
    """ Flujo de entramiento, evaluación y selección de modelos."""
    print("INICIANDO PIPELINE")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carga y división de datos
    print("1. Cargando dataset y separación 80/20")
    X_train, X_test, y_train, y_test = get_train_test_data()
    print(f"---- Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

    # 2. Preprocesamiento y guardado del preprocesador
    print("\n2. Aplicando limpieza, escaladao y OneHotEncoder")

    X_train_prep, X_test_prep, preprocessor = preprocess_data(X_train, X_test)

    prep_path = config.MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, prep_path)
    print(f"---- Preprocesador guardado en: {prep_path}")

    # 3. Cargar modelos disponibles
    print("\n3. Carga de modelos disponibles")
    models = load_available_models(tune=tune, X_train=X_train_prep, y_train=y_train)
    print(f"---- Modelos listos: {list(models.keys())}")

    # 4. Entrenamiento y Evaluación
    print("\n4. Entrenando y evaluando modelos")
    results = []
    trained_models = {}
    test_probas = {}

    for name, model in models.items():
        print(f"\n---- Entrenando: {name}")

        if name.startswith("Red Neuronal"):
            # El modelo procedente de tune_model ya está entrenado.
            if not getattr(model, "_already_trained", False):
                from src.models.neural_network import (
                    get_callbacks,
                    plot_training_history,
                )
                X_train_nn = np.asarray(
                    X_train_prep,
                    dtype=np.float32,
                )
                y_train_nn = np.asarray(
                    y_train,
                    dtype=np.float32,
                )
                history = model.fit(
                    X_train_nn,
                    y_train_nn,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=256,
                    callbacks=get_callbacks(patience=10),
                    verbose=1,
                    shuffle=True,
                )
                plot_training_history(history)

            X_test_nn = np.asarray(
                X_test_prep,
                dtype=np.float32,
            )
            y_proba = model.predict(
                X_test_nn,
                batch_size=256,
                verbose=0,
            ).ravel()
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            # Los modelos obtenidos mediante tuning ya están entrenados.
            if not hasattr(model, "n_features_in_"):
                model.fit(X_train_prep, y_train)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_prep)[:, 1]
                y_pred = model.predict(X_test_prep)
            else:
                y_proba = model.predict(X_test_prep).ravel()
                y_pred = (y_proba >= 0.5).astype(int)

        # Calculo de métricas
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["model"] = name
        results.append(metrics)
        trained_models[name] = model
        test_probas[name] = y_proba

        print(f"        Accuracy: {metrics['accuracy']:.4f}")
        print(f"        F1-Score: {metrics['f1_score']:.4f}")
        print(f"        ROC-AUC: {metrics['roc_auc']:.4f}")

    # 5. Tabla comparativa y selección del mejor modelo
    print("\n5. Generando comparativa y seleccionando mejor...")

    df_results = pd.DataFrame(results)

    cols = ["model", "roc_auc", "f1_score", "accuracy", "precision", "recall"]
    df_results = df_results[cols].sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    summary_path = config.REPORTS_DIR / "metrics_summary.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"\n---- Tabla comparativa: {summary_path}")
    print(df_results.to_string(index=False))

    best_name = df_results.iloc[0]["model"]
    best_auc = df_results.iloc[0]["roc_auc"]
    best_model = trained_models[best_name]

    best_model_path = config.MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    print(f"---- Modelo ganador en: {best_model_path}")

    # 6. Graficas del modelo ganador
    print("\n6. Generando gráficos de evaluación...")
    best_proba = test_probas[best_name]
    best_pred = (best_proba >= 0.5).astype(int)

    plot_confusion_matrix(y_test, best_pred, model_name=best_name)
    plot_roc_curve(y_test, best_proba, model_name=best_name)

    # Si es modelo basado en arboles
    if hasattr(best_model, "feature_importances_"):
        feature_names = preprocessor.get_feature_names_out()
        plot_feature_importance(best_model, feature_names, top_n=15, model_name=best_name)

    print("\n Pipeline completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento y evaluación")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    run_pipeline(tune=True)