# flake8: noqa: E501
import gzip
import json
import os
import pickle

import pandas as pd  # type: ignore

# ------------------------------------------------------------------------------
MODEL_PATH = "files/models/model.pkl.gz"
REQUIRED_COMPONENTS = [
    "OneHotEncoder",
    "SelectKBest",
    "MinMaxScaler",
    "LinearRegression",
]
SCORE_LIMITS = [
    -1.590,
    -2.429,
]
METRIC_LIMITS = [
    {
        "type": "metrics",
        "dataset": "train",
        "r2": 0.889,
        "mse": 5.950,
        "mad": 1.600,
    },
    {
        "type": "metrics",
        "dataset": "test",
        "r2": 0.728,
        "mse": 32.910,
        "mad": 2.430,
    },
]


# ------------------------------------------------------------------------------
#
# Internal tests
#
def _cargar_modelo():
    assert os.path.exists(MODEL_PATH)
    with gzip.open(MODEL_PATH, "rb") as archivo_modelo:
        modelo = pickle.load(archivo_modelo)
    assert modelo is not None
    return modelo


def _verificar_componentes(modelo):
    assert "GridSearchCV" in str(type(modelo))
    pasos_actuales = [str(modelo.estimator[i]) for i in range(len(modelo.estimator))]
    for componente in REQUIRED_COMPONENTS:
        assert any(componente in paso for paso in pasos_actuales)


def _cargar_datos_calificacion():
    with open("files/grading/x_train.pkl", "rb") as archivo:
        x_train = pickle.load(archivo)

    with open("files/grading/y_train.pkl", "rb") as archivo:
        y_train = pickle.load(archivo)

    with open("files/grading/x_test.pkl", "rb") as archivo:
        x_test = pickle.load(archivo)

    with open("files/grading/y_test.pkl", "rb") as archivo:
        y_test = pickle.load(archivo)

    return x_train, y_train, x_test, y_test


def _probar_scores(modelo, x_train, y_train, x_test, y_test):
    assert modelo.score(x_train, y_train) < SCORE_LIMITS[0]
    assert modelo.score(x_test, y_test) < SCORE_LIMITS[1]


def _leer_metricas_archivo():
    assert os.path.exists("files/output/metrics.json")
    lista_metricas: list[dict] = []
    with open("files/output/metrics.json", "r", encoding="utf-8") as archivo:
        for linea in archivo:
            lista_metricas.append(json.loads(linea))
    return lista_metricas


def _comprobar_metricas(metricas):
    for indice in [0, 1]:
        assert metricas[indice]["type"] == METRIC_LIMITS[indice]["type"]
        assert metricas[indice]["dataset"] == METRIC_LIMITS[indice]["dataset"]
        assert metricas[indice]["r2"] > METRIC_LIMITS[indice]["r2"]
        assert metricas[indice]["mse"] < METRIC_LIMITS[indice]["mse"]
        assert metricas[indice]["mad"] < METRIC_LIMITS[indice]["mad"]


def test_homework():

    modelo_entrenado = _cargar_modelo()
    x_train, y_train, x_test, y_test = _cargar_datos_calificacion()
    metricas_generadas = _leer_metricas_archivo()

    _verificar_componentes(modelo_entrenado)
    _probar_scores(modelo_entrenado, x_train, y_train, x_test, y_test)
    _comprobar_metricas(metricas_generadas)
