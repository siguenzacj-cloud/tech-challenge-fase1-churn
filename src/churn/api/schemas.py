"""Schemas pydantic da API (entrada e saída).

O schema de entrada é gerado a partir de `churn.config` (NUMERIC_FEATURES / CATEGORICAL_VALUES):
se as features ou os valores aceitos mudarem no config, a validação acompanha automaticamente.

Os nomes dos campos no JSON são os nomes originais das colunas do dataset (com espaços), que é
o que `churn.predict` espera. Dentro do Python, cada campo usa um nome sem espaços (alias).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from churn.config import CATEGORICAL_VALUES, NUMERIC_FEATURES

MAX_BATCH_SIZE = 1000

# Exibido como exemplo na documentação automática (/docs).
EXAMPLE_CUSTOMER: dict[str, Any] = {
    "Tenure Months": 2,
    "Monthly Charges": 70.7,
    "Total Charges": 151.65,
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
}


def _slug(column: str) -> str:
    return column.lower().replace(" ", "_")


def _build_customer_model() -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        fields[_slug(column)] = (float, Field(..., alias=column, ge=0))
    for column, values in CATEGORICAL_VALUES.items():
        fields[_slug(column)] = (Literal[tuple(values)], Field(..., alias=column))
    return create_model(
        "Customer",
        __config__=ConfigDict(
            populate_by_name=True,
            extra="forbid",
            json_schema_extra={"example": EXAMPLE_CUSTOMER},
        ),
        __module__=__name__,
        **fields,
    )


Customer = _build_customer_model()


class Prediction(BaseModel):
    probability: float = Field(..., ge=0, le=1, description="Probabilidade de churn")
    segment: str = Field(..., description="Segmento de risco")
    action: str = Field(..., description="Ação de retenção sugerida")


class CustomerBatch(BaseModel):
    customers: list[Customer] = Field(  # type: ignore[valid-type]
        ..., min_length=1, max_length=MAX_BATCH_SIZE
    )


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
