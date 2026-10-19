"""Pré-processamento das features."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Escala as numéricas e aplica one-hot nas categóricas.

    `handle_unknown="ignore"`: uma categoria nunca vista no treino não quebra a predição
    (todas as colunas dessa variável ficam em 0).
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
