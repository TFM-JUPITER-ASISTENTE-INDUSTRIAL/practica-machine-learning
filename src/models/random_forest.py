from sklearn.ensemble import RandomForestClassifier
from src import config



def get_model(
    random_state: int = config.RANDOM_STATE,
    **kwargs,
) -> RandomForestClassifier:
    
    # 1.Construye y devuelve un Random Forest sin entrenar.
    params = {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": random_state,
        "n_jobs": -1,
    }

    params.update(kwargs)

    return RandomForestClassifier(**params)



if __name__ == "__main__":
    from src.data_loader import get_train_test_data
    from src.evaluate import (
        compute_metrics,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_roc_curve,
    )
    from src.preprocessor import preprocess_data

    # 1. Carga y división de los datos
    print("1. Cargando datos...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # 2. Preprocesamiento
    print("\n2. Preprocesando datos...")
    X_train_prep, X_test_prep, preprocessor = preprocess_data(
        X_train,
        X_test,
    )
    print(f"Variables después del preprocesamiento: {X_train_prep.shape[1]}")

    # 3. Construcción y entrenamiento
    print("\n3. Entrenando Random Forest...")
    model = get_model()
    model.fit(X_train_prep, y_train)

    # 4. Predicciones
    print("\n4. Realizando predicciones...")
    y_pred = model.predict(X_test_prep)
    y_proba = model.predict_proba(X_test_prep)[:, 1]

    # 5. Cálculo de métricas
    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    print("\n5. Resultados del Random Forest baseline")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")


    # 6. Gráficos de evaluación
    print("\n6. Generando gráficos...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        model_name="Random Forest Baseline",
    )
    plot_roc_curve(
        y_test,
        y_proba,
        model_name="Random Forest Baseline",
    )
    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(
        model,
        feature_names,
        top_n=15,
        model_name="Random Forest Baseline",
    )
    print("\nEvaluación finalizada.")