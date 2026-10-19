"""Predição: probabilidade de churn → segmento de risco → ação de negócio.

Independente da API: recebe e devolve estruturas simples (dict / DataFrame), para ser usado
pela API, por um processamento em lote ou por testes.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import FEATURES, MODEL_PATH, SEGMENTS, Segment


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Carrega o pipeline treinado (pré-processamento + modelo)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. Execute `python -m churn.train` para gerá-lo."
        )
    return joblib.load(path)


def assign_segment(probability: float) -> Segment:
    """Retorna o segmento de risco correspondente à probabilidade de churn."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probabilidade fora do intervalo [0, 1]: {probability}")
    return [s for s in SEGMENTS if probability >= s.min_probability][-1]


def predict_customers(model: Pipeline, customers: pd.DataFrame) -> pd.DataFrame:
    """Prevê vários clientes de uma vez: probabilidade, segmento e ação para cada um."""
    missing = [col for col in FEATURES if col not in customers.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    probabilities = model.predict_proba(customers[FEATURES])[:, 1]
    segments = [assign_segment(p) for p in probabilities]
    return pd.DataFrame(
        {
            "probability": probabilities,
            "segment": [s.name for s in segments],
            "action": [s.action for s in segments],
        },
        index=customers.index,
    )


def predict_customer(model: Pipeline, customer: dict[str, Any]) -> dict[str, Any]:
    """Prevê um único cliente (dict com as 19 features, nomes iguais aos do dataset)."""
    result = predict_customers(model, pd.DataFrame([customer])).iloc[0]
    return {
        "probability": float(result["probability"]),
        "segment": result["segment"],
        "action": result["action"],
    }
