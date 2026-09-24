""" Módulo del modelo de Regresión Logística """

from sklearn.linear_model import LogisticRegression
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

import matplotlib.pyplot as plt

from src import config




def get_model():
    """ Construye y devuelve el modelo de Regresión Logística """

    modelo_rl = LogisticRegression(
        max_iter=200,
        random_state=config.RANDOM_STATE
    )

    return modelo_rl


def tune_model(X_train, y_train):
    """ Busca los mejores hiperparámetros mediante GridSearchCV """

    dict_parametros = {
        "C": [0.01, 0.1, 1],
        "max_iter": [100, 200],
        "solver": ["liblinear"]
    }

    modelo_rl = LogisticRegression(
        random_state=config.RANDOM_STATE
    )

    modelo_rl_cv = GridSearchCV(
        modelo_rl,
        dict_parametros,
        cv=5,
        scoring="accuracy"
    )

    modelo_rl_cv.fit(X_train, y_train)

    print(f"Mejores hiperparámetros: {modelo_rl_cv.best_params_}")
    print(f"Mejor score: {modelo_rl_cv.best_score_:.2%}")

    return modelo_rl_cv.best_estimator_


if __name__ == "__main__":
    from src.data_loader import get_train_test_data
    from src.preprocessor import preprocess_data

    # 1. Cargamos los datos
    X_train, X_test, y_train, y_test = get_train_test_data()

    # 2. Preprocesamos los datos
    X_train_prep, X_test_prep, _ = preprocess_data(X_train, X_test)

    # 3. Creamos y entrenamos el modelo
    modelo_rl = get_model()
    modelo_rl.fit(X_train_prep, y_train)

    # 4. Realizamos las predicciones
    y_pred = modelo_rl.predict(X_test_prep)

    # 5. Probabilidades de cancelación para ROC-AUC
    y_proba = modelo_rl.predict_proba(X_test_prep)[:, 1]
    # 6. calculamos valores de la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)


    # 7 Evaluamos el modelo
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
    plt.title("Matriz de Confusión - Regresión Logística")
    plt.show()
    

    # 8. Representamos curva ROC
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC - Regresión Logística")
    plt.show()

    #-------------------------------------------------------------------------------------------------------------------------------------
    # 9. Buscamos el mejor modelo con GridSearchCV
    modelo_rl_optimizado = tune_model(X_train_prep, y_train)

    # 10. Realizamos predicciones con el modelo optimizado
    y_pred_optimizado = modelo_rl_optimizado.predict(X_test_prep)

    # 11. Calculamos probabilidades del modelo optimizado
    y_proba_optimizado = modelo_rl_optimizado.predict_proba(X_test_prep)[:, 1]

    # 12. Evaluamos el modelo optimizado
    print("\n--- MODELO OPTIMIZADO ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_optimizado):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred_optimizado):.2%}")
    print(f"Recall: {recall_score(y_test, y_pred_optimizado):.2%}")
    print(f"F1-Score: {f1_score(y_test, y_pred_optimizado):.2%}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_optimizado):.2%}")