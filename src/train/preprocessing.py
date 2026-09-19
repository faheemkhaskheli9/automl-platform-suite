"""Automated preprocessing for the Train feature (issue #6): impute,
encode, and scale a dataset's feature columns before it reaches an
estimator, without hand-written per-dataset preprocessing code.

Pure pandas + scikit-learn, no Django import, mirroring `task_type.py`'s
framework-free design.

Strategy (documented per the acceptance criteria):
- Numeric columns: median imputation, then standard scaling.
- Categorical (object/category) columns: most-frequent imputation, then
  one-hot encoding (`handle_unknown="ignore"` so a category unseen at fit
  time is encoded as all-zero rather than raising at inference).

The pipeline is always *fit* on the training split only and only ever
*transforms* the test split -- fitting the scaler/encoder on the full
dataset would leak test-set statistics (e.g. the test set's mean/variance,
or its category frequencies) into training, silently inflating reported
performance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "PreprocessingSplit",
    "build_preprocessing_pipeline",
    "split_and_preprocess",
]


@dataclass(frozen=True)
class PreprocessingSplit:
    """A train/test split with preprocessing fit on the training half only."""

    pipeline: ColumnTransformer
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


def build_preprocessing_pipeline(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[ColumnTransformer, tuple[str, ...], tuple[str, ...]]:
    """Build a `ColumnTransformer` for `feature_columns`, split by dtype.

    Returns the (unfitted) pipeline plus the numeric/categorical column
    names it was built for, so a caller can report what happened without
    re-deriving the split.
    """
    numeric_columns = tuple(
        c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])
    )
    categorical_columns = tuple(c for c in feature_columns if c not in numeric_columns)

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))

    if not transformers:
        raise ValueError("feature_columns must be non-empty")

    pipeline = ColumnTransformer(transformers=transformers)
    return pipeline, numeric_columns, categorical_columns


def split_and_preprocess(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 0,
) -> PreprocessingSplit:
    """Split `df` into train/test, then fit preprocessing on the training
    split only and transform both splits with the fitted pipeline.
    """
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline, numeric_columns, categorical_columns = build_preprocessing_pipeline(
        df, feature_columns
    )
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    return PreprocessingSplit(
        pipeline=pipeline,
        X_train=np.asarray(X_train_transformed),
        X_test=np.asarray(X_test_transformed),
        y_train=y_train,
        y_test=y_test,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
