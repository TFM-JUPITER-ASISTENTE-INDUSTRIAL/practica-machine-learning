""" Módulo del modelo de Árbol de Decisión """

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src import config


def get_model():
    """ Construye el modelo base de Árbol de Decisión """

    modelo_dt = DecisionTreeClassifier(
        max_depth=3,
        random_state=config.RANDOM_STATE
    )

    return modelo_dt


def tune_model(X_train, y_train):
    """ Optimiza los hiperparámetros mediante GridSearchCV """

    dict_parametros = {
        "max_depth": [2, 3, 5, 10, 20, None],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4]
    }

    modelo_dt = DecisionTreeClassifier(
        random_state=config.RANDOM_STATE
    )

    modelo_dt_cv = GridSearchCV(
        modelo_dt,
        dict_parametros,
        cv=5,
        scoring="accuracy"
    )

    modelo_dt_cv.fit(X_train, y_train)

    print(f"Mejores hiperparámetros: {modelo_dt_cv.best_params_}")
    print(f"Mejor score: {modelo_dt_cv.best_score_:.2%}")

    return modelo_dt_cv.best_estimator_


if __name__ == "__main__":

    from src.data_loader import get_train_test_data
    from src.preprocessor import preprocess_data

    # Carga y preprocesamiento
    X_train, X_test, y_train, y_test = get_train_test_data()

    X_train_prep, X_test_prep, preprocessor = preprocess_data(
        X_train,
        X_test
    )

    # Modelo base
    modelo_dt = get_model()
    modelo_dt.fit(X_train_prep, y_train)

    y_pred = modelo_dt.predict(X_test_prep)
    y_proba = modelo_dt.predict_proba(X_test_prep)[:, 1]

    # Métricas
    print("--- MODELO BASE: ÁRBOL DE DECISIÓN ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred):.2%}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2%}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No cancela", "Cancela"]
    )

    disp.plot()
    plt.title("Matriz de Confusión - Árbol de Decisión")
    plt.show()

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC - Árbol de Decisión")
    plt.show()

    # Visualización del árbol
    feature_names = preprocessor.get_feature_names_out()

    plt.figure(figsize=(18, 10))
    plot_tree(
        modelo_dt,
        feature_names=feature_names,
        class_names=["No cancela", "Cancela"],
        filled=True
    )
    plt.title("Árbol de Decisión")
    plt.show()

    # Optimización con GridSearchCV
    modelo_dt_optimizado = tune_model(
        X_train_prep,
        y_train
    )

    y_pred_optimizado = modelo_dt_optimizado.predict(X_test_prep)
    y_proba_optimizado = modelo_dt_optimizado.predict_proba(X_test_prep)[:, 1]

    # Métricas modelo optimizado
    print("\n--- MODELO OPTIMIZADO: ÁRBOL DE DECISIÓN ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_optimizado):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred_optimizado):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred_optimizado):.2%}")
    print(f"F1-Score: {f1_score(y_test, y_pred_optimizado):.2%}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_optimizado):.2%}")