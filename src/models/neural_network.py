
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from src import config


    #####Construye y compila una red neuronal para clasificación binaria.
def get_model(
    input_dim: int,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2,
    random_state: int = config.RANDOM_STATE,
) -> tf.keras.Model:
    # Fijamos la semilla para mejorar la reproducibilidad.
    tf.keras.utils.set_random_seed(random_state)

    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input"),
            layers.Dense(128, activation="relu", name="hidden_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            layers.Dense(64, activation="relu", name="hidden_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),
            layers.Dense(32, activation="relu", name="hidden_3"),
            layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="hotel_cancellation_mlp",
    )
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


####Devuelve los callbacks utilizados durante el entrenamiento.
def get_callbacks(patience: int = 10) -> list:
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    return [early_stopping]


#####Guarda las curvas de aprendizaje de la red neuronal.
def plot_training_history(history) -> None:
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "training_history_neural_network.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))


    # Evolución de la función de pérdida.
    axes[0].plot(
        history.history["loss"],
        label="Train",
    )
    axes[0].plot(
        history.history["val_loss"],
        label="Validation",
    )
    axes[0].set_title("Evolución de la función de pérdida")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Binary crossentropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)


    # Evolución del AUC.
    axes[1].plot(
        history.history["auc"],
        label="Train",
    )
    axes[1].plot(
        history.history["val_auc"],
        label="Validation",
    )
    axes[1].set_title("Evolución del ROC-AUC")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)


    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    print(f"Curvas de aprendizaje guardadas en: {figure_path}")




if __name__ == "__main__":
    from src.data_loader import get_train_test_data
    from src.evaluate import (
        compute_metrics,
        plot_confusion_matrix,
        plot_roc_curve,
    )
    from src.preprocessor import preprocess_data

    # 1. Carga de los datos.
    print("1. Cargando datos...")
    X_train, X_test, y_train, y_test = get_train_test_data()

    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # 2. Preprocesamiento.
    print("\n2. Preprocesando datos...")
    X_train_prep, X_test_prep, _ = preprocess_data(
        X_train,
        X_test,
    )

    # TensorFlow trabaja más eficientemente con float32.
    X_train_prep = np.asarray(X_train_prep, dtype=np.float32)
    X_test_prep = np.asarray(X_test_prep, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32)
    y_test_array = np.asarray(y_test, dtype=np.float32)

    input_dim = X_train_prep.shape[1]
    print(f"Variables de entrada: {input_dim}")


    # 3. Construcción de la red.
    print("\n3. Construyendo red neuronal...")
    model = get_model(input_dim=input_dim)
    model.summary()

    # 4. Entrenamiento.
    print("\n4. Entrenando red neuronal...")
    history = model.fit(
        X_train_prep,
        y_train_array,
        validation_split=0.2,
        epochs=100,
        batch_size=256,
        callbacks=get_callbacks(patience=10),
        verbose=1,
        shuffle=True,
    )
    print(f"\nÉpocas ejecutadas: {len(history.history['loss'])}")

    # 5. Predicciones.
    print("\n5. Realizando predicciones...")
    y_proba = model.predict(
        X_test_prep,
        batch_size=256,
        verbose=0,
    ).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    # 6. Evaluación.
    metrics = compute_metrics(
        y_true=y_test_array,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    print("\n6. Resultados de la red neuronal")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    # 7. Gráficos.
    print("\n7. Generando gráficos...")
    plot_confusion_matrix(
        y_test_array,
        y_pred,
        model_name="Neural Network",
    )
    plot_roc_curve(
        y_test_array,
        y_proba,
        model_name="Neural Network",
    )
    plot_training_history(history)

    # 8. Guardado del modelo.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / "neural_network.keras"
    model.save(model_path)

    print(f"Modelo guardado en: {model_path}")
    print("\nEvaluación finalizada.")