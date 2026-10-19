"""Treina o modelo campeão e salva o modelo + metadados em models/.

Uso:
    python -m churn.train

Os hiperparâmetros vêm de `config.MODEL_PARAMS` (escolhidos na Etapa 2). A busca de
hiperparâmetros fica documentada em notebooks/02_modelagem.ipynb (experimentação).
"""

import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from churn.config import (
    CATEGORICAL_FEATURES,
    CUTOFFS,
    CV_FOLDS,
    METADATA_PATH,
    MODEL_NAME,
    MODEL_PARAMS,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    SEED,
    SEGMENTS,
)
from churn.data import load_dataset
from churn.features import build_preprocessor


def build_model() -> Pipeline:
    """Pipeline completo: pré-processamento + Regressão Logística."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(**MODEL_PARAMS)),
    ])


def evaluate(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Métricas usadas no projeto (ver Etapa 2)."""
    return {
        "AUC-ROC": round(roc_auc_score(y_true, proba), 3),
        "Brier": round(brier_score_loss(y_true, proba), 3),
        "Recall corte 50%": round(recall_score(y_true, (proba >= 0.5).astype(int)), 3),
        "Recall segmentação": round(recall_score(y_true, (proba >= CUTOFFS[0]).astype(int)), 3),
    }


def build_metadata(metrics_cv: dict, metrics_test: dict) -> dict:
    """Metadados do modelo: registro de como foi treinado e com quais regras de negócio."""
    return {
        "modelo": MODEL_NAME,
        "data_treinamento": date.today().isoformat(),
        "versao_sklearn": sklearn.__version__,
        "seed": SEED,
        "hiperparametros": MODEL_PARAMS,
        "features": {"numericas": NUMERIC_FEATURES, "categoricas": CATEGORICAL_FEATURES},
        "segmentacao": {
            "cortes": CUTOFFS,
            "segmentos": [
                {"nome": s.name, "probabilidade_minima": s.min_probability, "acao": s.action}
                for s in SEGMENTS
            ],
        },
        "metricas": {"validacao_cruzada": metrics_cv, "teste": metrics_test},
    }


def train() -> dict:
    """Treina, avalia (validação cruzada no treino + teste) e salva os artefatos."""
    X_train, X_test, y_train, y_test = load_dataset()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    proba_cv = cross_val_predict(build_model(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    model = build_model().fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]

    metadata = build_metadata(evaluate(y_train, proba_cv), evaluate(y_test, proba_test))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    resultado = train()
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Metadados salvos em: {METADATA_PATH}")
    for conjunto, metricas in resultado["metricas"].items():
        print(f"{conjunto}: {metricas}")
