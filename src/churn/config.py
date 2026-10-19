"""Configuração central do projeto — fonte única de verdade.

Rotas, semente, features, hiperparâmetros do modelo campeão e regras de segmentação de risco
ficam definidos apenas aqui. Os demais módulos importam daqui; nenhum valor é repetido.
"""

from dataclasses import dataclass
from pathlib import Path

# --- Rotas (relativas à raiz do repositório, independentes do diretório de execução) ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "Telco_customer_churn.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "champion_model.joblib"
METADATA_PATH = MODELS_DIR / "champion_metadata.json"

# --- Reprodutibilidade ---
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Dados ---
TARGET = "Churn Value"

NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]

# Valores aceitos por variável categórica (os mesmos encontrados no dataset de treino)
CATEGORICAL_VALUES = {
    "Gender": ["Female", "Male"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Multiple Lines": ["No", "No phone service", "Yes"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "No internet service", "Yes"],
    "Online Backup": ["No", "No internet service", "Yes"],
    "Device Protection": ["No", "No internet service", "Yes"],
    "Tech Support": ["No", "No internet service", "Yes"],
    "Streaming TV": ["No", "No internet service", "Yes"],
    "Streaming Movies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["No", "Yes"],
    "Payment Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}
CATEGORICAL_FEATURES = list(CATEGORICAL_VALUES)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Modelo campeão (escolhido na Etapa 2 — ver notebooks/02_modelagem.ipynb) ---
MODEL_NAME = "Regressão Logística ajustada"
MODEL_PARAMS = {
    "C": 100,
    "l1_ratio": 0,          # 0 = regularização L2
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": SEED,
}


# --- Segmentação de risco (decisão de negócio da Etapa 2) ---
@dataclass(frozen=True)
class Segment:
    """Faixa de risco: clientes com probabilidade >= min_probability (até a próxima faixa)."""

    name: str
    min_probability: float
    action: str


# Ordenados da menor para a maior probabilidade
SEGMENTS = (
    Segment(
        name="Acompanhamento",
        min_probability=0.0,
        action="Nenhuma ação específica.",
    ),
    Segment(
        name="Retenção conservadora",
        min_probability=0.20,
        action=(
            "Gestão comercial para contratação de produtos de barreira de saída: "
            "mudança do tipo de contrato, do tipo de serviço, inclusão de dependentes."
        ),
    ),
    Segment(
        name="Retenção agressiva",
        min_probability=0.50,
        action="Produtos de barreira de saída com bonificação de 1 mês de serviço grátis.",
    ),
)

# Cortes derivados dos segmentos (não editar aqui: altere SEGMENTS)
CUTOFFS = [s.min_probability for s in SEGMENTS[1:]]
