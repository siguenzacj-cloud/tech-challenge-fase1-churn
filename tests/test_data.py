import pandas as pd
from pandas.api.types import is_numeric_dtype

from churn.config import FEATURES, TARGET
from churn.data import (
    fix_total_charges,
    split_features_target,
    split_train_test,
)


def test_fix_total_charges_converts_text_and_blanks_to_numeric() -> None:
    df = pd.DataFrame(
        {
            "Total Charges": [
                "151,65",
                " ",
                "",
                None,
                "200.50",
            ]
        }
    )

    result = fix_total_charges(df)

    assert result["Total Charges"].tolist() == [
        151.65,
        0.0,
        0.0,
        0.0,
        200.5,
    ]
    assert is_numeric_dtype(result["Total Charges"])


def test_fix_total_charges_preserves_original_dataframe() -> None:
    df = pd.DataFrame(
        {
            "Total Charges": [
                "151,65",
                " ",
            ]
        }
    )

    original = df.copy(deep=True)

    fix_total_charges(df)

    pd.testing.assert_frame_equal(df, original)


def test_split_features_target_uses_only_configured_features() -> None:
    data = {column: [0] for column in FEATURES}
    data[TARGET] = [1]
    data["Churn Score"] = [99]

    df = pd.DataFrame(data)

    X, y = split_features_target(df)

    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert "Churn Score" not in X.columns


def test_split_train_test_is_reproducible_and_stratified() -> None:
    X = pd.DataFrame({"feature": range(20)})
    y = pd.Series([0, 1] * 10, name=TARGET)

    first_split = split_train_test(X, y)
    second_split = split_train_test(X, y)

    X_train_1, X_test_1, y_train_1, y_test_1 = first_split
    X_train_2, X_test_2, y_train_2, y_test_2 = second_split

    assert X_train_1.index.tolist() == X_train_2.index.tolist()
    assert X_test_1.index.tolist() == X_test_2.index.tolist()
    assert y_train_1.index.tolist() == y_train_2.index.tolist()
    assert y_test_1.index.tolist() == y_test_2.index.tolist()

    assert y_train_1.value_counts().to_dict() == {0: 8, 1: 8}
    assert y_test_1.value_counts().to_dict() == {0: 2, 1: 2}
