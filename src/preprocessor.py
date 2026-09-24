""" Módulo para preprocesamiento """
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

def clean_features(df):
    """ Aplica limipeza inicial y de caracteristicas del EDA """
    X = df.copy()

    # Eliminamos los Undefined detectados
    if "market_segment" in X.columns:
        X["market_segment"] = X["market_segment"].replace("Undefined", "Online TA")
    if "distribution_channel" in X.columns:
        X["distribution_channel"] = X["distribution_channel"].replace("Undefined", "TA/TO")
    if "meal" in X.columns:
        X["meal"] = X["meal"].replace("Undefined", "SC")

    # Convertimos company en booleano
    if "company" in X.columns:
        X["has_company"] = X["company"].notnull().astype(int)
        X = X.drop(columns=["company"])

    # Eliminamos nulos en agent y los convertimos en 0
    if "agent" in X.columns:
        X["agent"] = X["agent"].fillna(0)
        X["agent"] = X["agent"].apply(
            lambda a: str(a) if a in config.TOP_AGENTS or a == 0 else "Other"
        )

    # Eliminamos nulos en children y los convertimos en 0
    if "children" in X.columns:
        X["children"] = X["children"].fillna(0)

    # Dejamos el TOP 5 de paises y el resto los agrupamos en Other
    if "country" in X.columns:
        X["country"] = X["country"].fillna("Other")
        X["country"] = X["country"].apply(lambda c: c if c in config.TOP_COUNTRIES else "Other")

    return X


def get_preprocessor():
    """ Construye el ColumnTransformer de scikit-learn con StandardScaler y OneHotEncoder."""
    # Variables numéricas: las estándar + has_company
    numeric_features = config.NUMERICAL_COLS + ["has_company"]
    categorical_features = config.CATEGORICAL_COLS + ["agent"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_features),
        ]
    )
    return preprocessor

def preprocess_data(X_train, X_test):
    """ Limpia y transforma los datos de Train y Test """
    X_train_clean = clean_features(X_train)
    X_test_clean = clean_features(X_test)

    preprocessor = get_preprocessor()

    X_train_prep = preprocessor.fit_transform(X_train_clean)
    X_test_prep = preprocessor.transform(X_test_clean)

    return X_train_prep, X_test_prep, preprocessor