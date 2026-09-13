""" Módulo de carga de datos y partición de entrenamiento / prueba """
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

def load_raw_data():
    """ Carga el dataset"""
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"{config.RAW_DATA_PATH} no existe")
    return pd.read_csv(config.RAW_DATA_PATH)

def get_train_test_data():
    """ Carga, limpia y devuelve train y test"""
    df = load_raw_data()
    df_clean = df.drop(columns=config.LEAKAGE_COLS)

    X = df_clean.drop(columns=config.TARGET_COL)
    y = df_clean[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify = y
    )
    return X_train, X_test, y_train, y_test