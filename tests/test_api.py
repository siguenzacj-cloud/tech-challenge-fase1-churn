"""Testes da API de inferência. Requer o modelo gerado por `python -m churn.train`."""

import pytest
from fastapi.testclient import TestClient

from churn.api.main import app
from churn.config import CATEGORICAL_VALUES, MODEL_PATH, SEGMENTS

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Modelo não encontrado. Execute `python -m churn.train` antes dos testes.",
)

CUSTOMER = {
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


@pytest.fixture(scope="module")
def client():
    # O `with` executa o lifespan, que carrega o modelo.
    with TestClient(app) as c:
        yield c


def test_every_categorical_value_in_example_is_valid():
    for column, values in CATEGORICAL_VALUES.items():
        assert CUSTOMER[column] in values


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_probability_segment_and_action(client):
    response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["segment"] in {s.name for s in SEGMENTS}
    assert body["action"]


def test_predict_segment_matches_probability(client):
    body = client.post("/predict", json=CUSTOMER).json()
    expected = [s for s in SEGMENTS if body["probability"] >= s.min_probability][-1]
    assert body["segment"] == expected.name
    assert body["action"] == expected.action


def test_predict_rejects_invalid_category(client):
    response = client.post("/predict", json={**CUSTOMER, "Contract": "Weekly"})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in CUSTOMER.items() if k != "Tenure Months"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_negative_charges(client):
    response = client.post("/predict", json={**CUSTOMER, "Monthly Charges": -1})
    assert response.status_code == 422


def test_predict_rejects_unknown_field(client):
    response = client.post("/predict", json={**CUSTOMER, "Foo": "bar"})
    assert response.status_code == 422


def test_batch_returns_one_prediction_per_customer(client):
    other = {**CUSTOMER, "Contract": "Two year", "Tenure Months": 60}
    response = client.post("/predict/batch", json={"customers": [CUSTOMER, other]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    single = client.post("/predict", json=CUSTOMER).json()
    assert predictions[0]["probability"] == pytest.approx(single["probability"])


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
