""" Módulo del modelo XGBoost """
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from src import config

def get_model(random_state: int = config.RANDOM_STATE, **kwargs) -> XGBClassifier:
    """ Construye y devuelve instancia sin entrengar de XGBClassifier """
    params = {
        "n_estimators" : 150,
        "learning_rate" : 0.1,
        "max_depth" : 6,
        "random_state" : random_state,
        "n_jobs" : -1,
        "eval_metric" : "logloss",
    }
    params.update(kwargs)
    return XGBClassifier(**params)

def tune_model(X_train, y_train, n_iter: int = 8, cv: int = 3, random_state: int = config.RANDOM_STATE):
    """ Optimiza hiperparámetros buscando maximixar resultados """
    params_distributions = {
        "n_estimators": [100, 150, 200, 250, 300],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1],
    }

    base_model = get_model(random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=params_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    print(f"Iniciando tuning de XGBoost {n_iter} combinaciones, cv={cv}...")
    search.fit(X_train, y_train)

    print(f"\n Tuning completado")
    print(f"Mejor ROC-AUC en CV: {search.best_score_:.4f}")
    print(f"Mejores hiperparámetros: {search.best_params_}")

    return search.best_estimator_, search.best_params_

if __name__ == "__main__":
    from src.data_loader import get_train_test_data
    from src.preprocessor import preprocess_data
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    # 1. Carga y preprocesamiento
    print("Cargando y procesando datos...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    X_train_prep, X_test_prep, _ = preprocess_data(X_train, X_test)

    # 2. Modelo Baseline (por defecto)
    print("\n--- 1. EVALUANDO XGBOOST BASELINE ---")
    base_model = get_model()
    base_model.fit(X_train_prep, y_train)
    y_pred_base = base_model.predict(X_test_prep)
    y_proba_base = base_model.predict_proba(X_test_prep)[:, 1]
    print(f"Baseline - Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"Baseline - F1-Score: {f1_score(y_test, y_pred_base):.4f}")
    print(f"Baseline - ROC-AUC:  {roc_auc_score(y_test, y_proba_base):.4f}")

    # 3. Modelo Optimizado (Tuning)
    print("\n--- 2. OPTIMIZANDO XGBOOST (TUNING) ---")
    best_model, best_params = tune_model(X_train_prep, y_train, n_iter=6, cv=3)

    # 4. Evaluación del modelo optimizado en el conjunto de Test
    y_pred_tuned = best_model.predict(X_test_prep)
    y_proba_tuned = best_model.predict_proba(X_test_prep)[:, 1]
    print("\n--- 3. RESULTADOS EN TEST TRAS TUNING ---")
    print(f"Tuned - Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
    print(f"Tuned - F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")
    print(f"Tuned - ROC-AUC:  {roc_auc_score(y_test, y_proba_tuned):.4f}")