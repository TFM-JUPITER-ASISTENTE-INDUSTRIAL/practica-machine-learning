
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from src import config
from sklearn.model_selection import train_test_split

    #####Construye y compila una red neuronal para clasificación binaria.
def get_model(
    input_dim: int,
    hidden_units: tuple = (128, 64, 32),
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2,
    random_state: int = config.RANDOM_STATE,
) -> tf.keras.Model:
    # Fijamos la semilla para mejorar la reproducibilidad.
    tf.keras.utils.set_random_seed(random_state)

    model_layers = [
        layers.Input(shape=(input_dim,), name="input"),
    ]

    # Construimos las capas ocultas según la configuración recibida.
    for index, units in enumerate(hidden_units, start=1):
        model_layers.append(
            layers.Dense(
                units,
                activation="relu",
                name=f"hidden_{index}",
            )
        )

        # Añadimos Dropout entre las capas ocultas.
        if index < len(hidden_units):
            model_layers.append(
                layers.Dropout(
                    dropout_rate,
                    name=f"dropout_{index}",
                )
            )

    # Salida para clasificación binaria.
    model_layers.append(
        layers.Dense(
            1,
            activation="sigmoid",
            name="output",
        )
    )

    model = models.Sequential(
        model_layers,
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



###Compara configuraciones y devuelve la mejor red neuronal.
def tune_model(
    X_train,
    y_train,
    max_epochs: int = 100,
    patience: int = 10,
    random_state: int = config.RANDOM_STATE,
):


    X_array = np.asarray(X_train, dtype=np.float32)
    y_array = np.asarray(y_train, dtype=np.float32)

    # Separamos una validación interna sin utilizar el test.
    X_tune_train, X_validation, y_tune_train, y_validation = (
        train_test_split(
            X_array,
            y_array,
            test_size=0.2,
            random_state=random_state,
            stratify=y_array,
        )
    )

    # Configuraciones basadas en los ejercicios de la sesión 4.
    configurations = [
        {
            "hidden_units": (64, 32),
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 256,
        },
        {
            "hidden_units": (128, 64, 32),
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 256,
        },
        {
            "hidden_units": (128, 64, 32),
            "dropout_rate": 0.3,
            "learning_rate": 0.0005,
            "batch_size": 256,
        },
    ]

    best_score = -np.inf
    best_params = None
    best_epoch = None

    for experiment, params in enumerate(configurations, start=1):
        print(
            f"\nExperimento {experiment}/"
            f"{len(configurations)}"
        )
        print(f"Parámetros: {params}")

        model = get_model(
            input_dim=X_array.shape[1],
            hidden_units=params["hidden_units"],
            dropout_rate=params["dropout_rate"],
            learning_rate=params["learning_rate"],
            random_state=random_state,
        )

        early_stopping = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )

        history = model.fit(
            X_tune_train,
            y_tune_train,
            validation_data=(X_validation, y_validation),
            epochs=max_epochs,
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
            verbose=1,
            shuffle=True,
        )

        validation_auc = max(history.history["val_auc"])

        selected_epoch = int(
            np.argmax(history.history["val_auc"]) + 1
        )

        print(f"Mejor val_auc: {validation_auc:.4f}")
        print(f"Mejor época: {selected_epoch}")

        if validation_auc > best_score:
            best_score = validation_auc
            best_params = params.copy()
            best_epoch = selected_epoch

    print("\nTuning de la red neuronal completado")
    print(f"Mejor ROC-AUC de validación: {best_score:.4f}")
    print(f"Mejores parámetros: {best_params}")
    print(f"Mejor número de épocas: {best_epoch}")

    # Reconstruimos y entrenamos el modelo ganador
    # utilizando todo el conjunto de entrenamiento.
    best_model = get_model(
        input_dim=X_array.shape[1],
        hidden_units=best_params["hidden_units"],
        dropout_rate=best_params["dropout_rate"],
        learning_rate=best_params["learning_rate"],
        random_state=random_state,
    )

    best_model.fit(
        X_array,
        y_array,
        epochs=best_epoch,
        batch_size=best_params["batch_size"],
        verbose=1,
        shuffle=True,
    )

    # train.py utilizará esta marca para no entrenarlo otra vez.
    best_model._already_trained = True

    best_params["epochs"] = best_epoch

    return best_model, best_params

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

    # 9. Optimización y evaluación de la red optimizada.
    print("\n9. Optimizando red neuronal...")

    best_model, best_params = tune_model(
        X_train_prep,
        y_train_array,
    )

    # tune_model ya devuelve la red entrenada.
    y_proba_tuned = best_model.predict(
        X_test_prep,
        batch_size=best_params["batch_size"],
        verbose=0,
    ).ravel()

    y_pred_tuned = (y_proba_tuned >= 0.5).astype(int)

    metrics_tuned = compute_metrics(
        y_test_array,
        y_pred_tuned,
        y_proba_tuned,
    )

    print("\nComparación en test: baseline / optimizado")
    for metric_name, baseline_value in metrics.items():
        print(
            f"{metric_name}: {baseline_value:.4f}"
            f" / {metrics_tuned[metric_name]:.4f}"
        )

    plot_confusion_matrix(
        y_test_array,
        y_pred_tuned,
        model_name="Neural Network Tuned",
    )
    plot_roc_curve(
        y_test_array,
        y_proba_tuned,
        model_name="Neural Network Tuned",
    )