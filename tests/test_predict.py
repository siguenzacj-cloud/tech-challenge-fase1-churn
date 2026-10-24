from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.config import (
    CATEGORICAL_VALUES,
    FEATURES,
    NUMERIC_FEATURES,
    SEGMENTS,
)
from churn.predict import (
    assign_segment,
    predict_customer,
    predict_customers,
)


class FakeModel:
    """Modelo simplificado para testar a lógica de predição sem depender do modelo treinado."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.array(
            self.probabilities[: len(X)],
            dtype=float,
        )

        return np.column_stack(
            [
                1 - probabilities,
                probabilities,
            ]
        )


def make_valid_customer() -> dict[str, Any]:
    customer: dict[str, Any] = {
        column: 1.0
        for column in NUMERIC_FEATURES
    }

    customer.update(
        {
            column: values[0]
            for column, values in CATEGORICAL_VALUES.items()
        }
    )

    return customer


@pytest.mark.parametrize(
    "probability, expected_segment",
    [
        (0.00, "Acompanhamento"),
        (0.19, "Acompanhamento"),
        (0.20, "Retenção conservadora"),
        (0.49, "Retenção conservadora"),
        (0.50, "Retenção agressiva"),
        (1.00, "Retenção agressiva"),
    ],
    ids=[
        "acompanhamento-0.00",
        "acompanhamento-0.19",
        "conservadora-0.20",
        "conservadora-0.49",
        "agressiva-0.50",
        "agressiva-1.00",
    ],
)
def test_assign_segment_respects_probability_boundaries(
    probability: float,
    expected_segment: str,
) -> None:
    segment = assign_segment(probability)

    assert segment.name == expected_segment


@pytest.mark.parametrize(
    "probability",
    [
        -0.01,
        1.01,
    ],
    ids=[
        "below-zero",
        "above-one",
    ],
)
def test_assign_segment_rejects_invalid_probability(
    probability: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Probabilidade fora do intervalo",
    ):
        assign_segment(probability)


def test_predict_customers_rejects_missing_features() -> None:
    customer = make_valid_customer()

    customers = pd.DataFrame([customer]).drop(
        columns=[FEATURES[0]]
    )

    model = cast(
        Pipeline,
        FakeModel([0.30]),
    )

    with pytest.raises(
        ValueError,
        match="Colunas ausentes",
    ):
        predict_customers(model, customers)


def test_predict_customers_returns_expected_structure() -> None:
    customers = pd.DataFrame(
        [
            make_valid_customer(),
            make_valid_customer(),
        ]
    )

    customers.index = [10, 20]

    model = cast(
        Pipeline,
        FakeModel(
            [
                0.10,
                0.70,
            ]
        ),
    )

    result = predict_customers(
        model,
        customers,
    )

    assert list(result.columns) == [
        "probability",
        "segment",
        "action",
    ]

    assert result.index.tolist() == [10, 20]

    assert result.loc[10, "probability"] == pytest.approx(0.10)
    assert result.loc[10, "segment"] == "Acompanhamento"

    assert result.loc[20, "probability"] == pytest.approx(0.70)
    assert result.loc[20, "segment"] == "Retenção agressiva"

    assert result.loc[10, "action"]
    assert result.loc[20, "action"]


def test_predict_customer_returns_probability_segment_and_action() -> None:
    customer = make_valid_customer()

    model = cast(
        Pipeline,
        FakeModel([0.35]),
    )

    result = predict_customer(
        model,
        customer,
    )

    assert result["probability"] == pytest.approx(0.35)
    assert result["segment"] == "Retenção conservadora"

    expected_segment = next(
        segment
        for segment in SEGMENTS
        if segment.name == "Retenção conservadora"
    )

    assert result["action"] == expected_segment.action