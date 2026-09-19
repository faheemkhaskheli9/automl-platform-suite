"""Target-column selection and task-type detection for the Train feature
(issue #5): pure pandas + stdlib, no Django import, mirroring
`automl_core.validation`'s framework-free design so this logic is testable
and reusable independent of any view.

Task-type values are exactly the `classification` / `regression` strings
`automl_core.metrics_schema` already defines -- this module is the only
place that *infers* one from data; every downstream consumer (Train's model
search, Evaluate's dashboard) keeps reading that single shared vocabulary
instead of a second, train-specific one (see robustness rule: duplicated
vocabularies need to stay in sync, so there's only one to begin with).

Design note (informed by the `configurable-model-training` pattern in the
LLM/ML knowledge base -- its target-spec design requires the task type to be
checked explicitly against the estimator rather than assumed): the same
principle applies here one step earlier, at target-column selection. A
heuristic that quietly guesses wrong (numeric-but-categorical read as
regression, or a free-text/ID column read as a 900-class classification
target) corrupts every downstream training run silently. So `detect_task_type`
refuses to guess for the ambiguous case and `analyze_target_column` requires
an explicit override before proceeding -- see robustness rule 5 in
CLAUDE.md ("prefer raising ... over guessing wrong silently").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from automl_core.metrics_schema import CLASSIFICATION, REGRESSION, TASK_TYPES

# Numeric columns with at most this many distinct whole-number values look
# like encoded class labels (e.g. 0/1/2), not a continuous target -- without
# this check a numeric-but-categorical column would silently score as
# regression.
NUMERIC_LOW_CARDINALITY_MAX_UNIQUE = 10

# Non-numeric columns where more than this fraction of non-null values are
# distinct look like an identifier/free-text column, not a classification
# target (e.g. a name/ID/free-text column) -- auto-detection refuses to
# guess here rather than returning "classification" with hundreds of
# "classes".
HIGH_CARDINALITY_MAX_RATIO = 0.5

__all__ = [
    "TargetColumnError",
    "TargetAnalysis",
    "detect_task_type",
    "analyze_target_column",
]


class TargetColumnError(ValueError):
    """Raised for anything about an explicitly chosen target column that
    would silently corrupt downstream training if allowed through: the
    column doesn't exist, it has too few distinct values to model, or it's
    too ambiguous for auto-detection to guess a task type for. Always raised
    loudly -- never swallowed into a best-effort guess.
    """


@dataclass(frozen=True)
class TargetAnalysis:
    """Result of successfully analyzing a target column."""

    column: str
    detected_task_type: str | None
    effective_task_type: str
    overridden: bool
    n_unique: int
    n_missing: int
    n_rows: int
    warnings: list[str] = field(default_factory=list)


def _looks_integer(series: pd.Series) -> bool:
    """True if every non-null value is numerically a whole number (handles
    float columns like `1.0, 2.0` -- common once a column has any nulls,
    which upcasts an otherwise-integer pandas column to float64)."""
    return bool(((series % 1) == 0).all())


def detect_task_type(non_null: pd.Series) -> tuple[str | None, str | None]:
    """Infer a task type from a target column's *non-null* values.

    Returns `(task_type, warning)`. `task_type` is `None` when the column is
    too ambiguous to guess confidently (high-cardinality text) -- callers
    must then require an explicit override rather than falling back to a
    default.
    """
    n = len(non_null)
    n_unique = int(non_null.nunique())

    if pd.api.types.is_bool_dtype(non_null):
        return CLASSIFICATION, None

    if pd.api.types.is_numeric_dtype(non_null):
        if n_unique <= NUMERIC_LOW_CARDINALITY_MAX_UNIQUE and _looks_integer(non_null):
            return CLASSIFICATION, None
        return REGRESSION, None

    # Non-numeric (object/category/string) column.
    ratio = (n_unique / n) if n else 0.0
    if ratio > HIGH_CARDINALITY_MAX_RATIO:
        return None, (
            f"{n_unique} distinct values across {n} rows ({ratio:.0%} unique) "
            "looks like an identifier or free-text column, not a "
            "classification target -- pick a task type explicitly to proceed."
        )
    return CLASSIFICATION, None


def analyze_target_column(
    df: pd.DataFrame,
    column: str,
    *,
    override_task_type: str | None = None,
) -> TargetAnalysis:
    """Validate and analyze a user-chosen target column.

    Hard errors (`TargetColumnError`) are raised, regardless of any
    override, for anything that would make the column unusable as a target
    at all: it doesn't exist, it has no non-null values, or it has fewer
    than 2 distinct values. An override only ever picks *which* task type to
    use for an otherwise-valid column -- it never rescues an invalid one.
    """
    if column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise TargetColumnError(
            f"Column {column!r} not found in the dataset. Available columns: {available}."
        )

    series = df[column]
    n_rows = len(series)
    non_null = series.dropna()
    n_missing = int(series.isna().sum())

    if non_null.empty:
        raise TargetColumnError(f"Target column {column!r} has no non-null values.")

    n_unique = int(non_null.nunique())
    if n_unique < 2:
        raise TargetColumnError(
            f"Target column {column!r} is degenerate: only {n_unique} distinct "
            "value(s) across the non-null rows, so no model could be trained on it."
        )

    warnings: list[str] = []
    if n_missing:
        warnings.append(
            f"{n_missing} row(s) have a missing value in {column!r}; those rows "
            "will need to be dropped before training."
        )

    detected, detect_warning = detect_task_type(non_null)
    if detect_warning:
        warnings.append(detect_warning)

    if override_task_type is not None:
        if override_task_type not in TASK_TYPES:
            raise TargetColumnError(
                f"Unknown task_type override {override_task_type!r} "
                f"(must be one of {', '.join(TASK_TYPES)})."
            )
        effective = override_task_type
    elif detected is not None:
        effective = detected
    else:
        raise TargetColumnError(
            f"Could not confidently infer a task type for {column!r} "
            f"({detect_warning}); pass an explicit task_type override to proceed."
        )

    return TargetAnalysis(
        column=column,
        detected_task_type=detected,
        effective_task_type=effective,
        overridden=override_task_type is not None,
        n_unique=n_unique,
        n_missing=n_missing,
        n_rows=n_rows,
        warnings=warnings,
    )
