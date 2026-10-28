"""API REST de inferência de churn (FastAPI).

Executar (a partir da raiz do repositório):
    uvicorn churn.api.main:app --reload
Documentação interativa: http://127.0.0.1:8000/docs
"""

import json
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from churn.api.schemas import BatchPrediction, Customer, CustomerBatch, Prediction
from churn.config import METADATA_PATH, MODEL_NAME
from churn.predict import load_model, predict_customer, predict_customers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma única vez, na inicialização."""
    app.state.model = load_model()
    yield


app = FastAPI(
    title="API de Predição de Churn",
    description=(
        "Recebe os dados de um cliente de telecom e devolve a probabilidade de churn, "
        "o segmento de risco e a ação de retenção sugerida."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return model


@app.get("/health", tags=["Monitoramento"])
def health(request: Request) -> dict[str, Any]:
    """Status da API e se o modelo está carregado."""
    metadata: dict[str, Any] = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "model", None) is not None,
        "model_name": MODEL_NAME,
        "metadata": metadata,
    }


@app.post("/predict", response_model=Prediction, tags=["Predição"])
def predict(customer: Customer, request: Request) -> Prediction:  # type: ignore[valid-type]
    """Prevê o risco de churn de um único cliente."""
    model = _get_model(request)
    return Prediction(**predict_customer(model, customer.model_dump(by_alias=True)))


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Predição"])
def predict_batch(batch: CustomerBatch, request: Request) -> BatchPrediction:
    """Prevê vários clientes de uma vez (até 1000)."""
    model = _get_model(request)
    rows = pd.DataFrame([c.model_dump(by_alias=True) for c in batch.customers])
    results = predict_customers(model, rows)
    return BatchPrediction(
        predictions=[
            Prediction(probability=float(r.probability), segment=r.segment, action=r.action)
            for r in results.itertuples()
        ]
    )
