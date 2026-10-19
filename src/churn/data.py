"""Carga, limpeza e separação dos dados (Telco Customer Churn)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import DATA_PATH, FEATURES, SEED, TARGET, TEST_SIZE


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lê o CSV original (separador ';' e decimal ',')."""
    return pd.read_csv(path, sep=";", decimal=",")


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Converte `Total Charges` para número.

    No dataset original a coluna vem como texto, com vírgula decimal e valores em branco para
    clientes com 0 meses de contrato (ainda não faturados). Os brancos viram 0.
    Retorna uma cópia; o DataFrame original não é alterado.
    """
    df = df.copy()
    texto = df["Total Charges"].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["Total Charges"] = pd.to_numeric(texto, errors="coerce").fillna(0.0)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features usadas pelo modelo (sem colunas com vazamento) e o target."""
    return df[FEATURES], df[TARGET]


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split 80/20 estratificado — idêntico ao usado nos notebooks."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)


def load_dataset(
    path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Atalho: carrega, limpa e separa em X_train, X_test, y_train, y_test."""
    df = fix_total_charges(load_raw_data(path))
    X, y = split_features_target(df)
    return split_train_test(X, y)
