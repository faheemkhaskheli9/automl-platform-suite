"""Automated preprocessing pipeline tests (issue #6).

Pure-logic tests against `train.preprocessing` -- no Django test client
needed, mirroring `test_task_type.py`.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from train.preprocessing import build_preprocessing_pipeline, split_and_preprocess


def _mixed_df(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "income": rng.normal(50_000, 5_000, n),
            "city": rng.choice(["ny", "sf", "la"], n),
            "label": rng.integers(0, 2, n),
        }
    )


def test_build_pipeline_splits_columns_by_dtype():
    df = _mixed_df()
    _, numeric_columns, categorical_columns = build_preprocessing_pipeline(
        df, ["age", "income", "city"]
    )
    assert set(numeric_columns) == {"age", "income"}
    assert set(categorical_columns) == {"city"}


def test_build_pipeline_requires_at_least_one_feature_column():
    df = _mixed_df()
    with pytest.raises(ValueError, match="non-empty"):
        build_preprocessing_pipeline(df, [])


def test_missing_numeric_values_are_imputed_not_dropped():
    df = _mixed_df()
    df.loc[0:5, "age"] = np.nan
    result = split_and_preprocess(df, ["age", "income", "city"], "label", random_state=0)
    assert not np.isnan(result.X_train).any()
    assert not np.isnan(result.X_test).any()


def test_missing_categorical_values_are_imputed():
    df = _mixed_df()
    df.loc[0:5, "city"] = None
    # Would raise inside OneHotEncoder/SimpleImputer if unhandled NaNs reached it.
    split_and_preprocess(df, ["age", "income", "city"], "label", random_state=0)


def test_categorical_column_is_one_hot_encoded():
    df = _mixed_df()
    result = split_and_preprocess(df, ["age", "income", "city"], "label", random_state=0)
    # 2 numeric columns + up to 3 one-hot city columns (all after fit on train split).
    assert result.X_train.shape[1] >= 2 + 2  # at least 2 of the 3 city categories appear


def test_numeric_columns_are_scaled_to_roughly_zero_mean():
    df = _mixed_df(n=200)
    result = split_and_preprocess(df, ["age", "income"], "label", random_state=0)
    assert abs(result.X_train.mean()) < 0.5


def test_pipeline_is_fit_on_training_split_only_no_leakage():
    df = _mixed_df(n=100)
    result = split_and_preprocess(df, ["age", "income"], "label", test_size=0.3, random_state=0)
    assert result.X_train.shape[0] == 70
    assert result.X_test.shape[0] == 30

    # Reproduce the same deterministic split to get the raw rows the
    # pipeline actually saw, and confirm the fitted scaler's mean_ matches
    # only those training rows -- not the full 100-row dataset (which would
    # mean test-set statistics leaked into the fitted preprocessing).
    X_train_raw, _X_test_raw, _y_train, _y_test = train_test_split(
        df[["age", "income"]], df["label"], test_size=0.3, random_state=0
    )
    fitted_scaler = result.pipeline.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(fitted_scaler.mean_, X_train_raw.mean().to_numpy())
    assert not np.allclose(fitted_scaler.mean_, df[["age", "income"]].mean().to_numpy())
