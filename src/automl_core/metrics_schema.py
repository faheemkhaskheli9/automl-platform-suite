"""Unified metrics schema shared by Train, Tune, and Evaluate (issue #3).

Framework-free (no Django import) so it can be validated/tested without a
database, and reused as-is by every feature app that produces or reads
metrics. `SCHEMA_VERSION` is recorded on every stored record (see
`MetricsResult` in `automl_core.models`) so a future format change can be
detected and migrated instead of silently misreading an old run.
"""
from __future__ import annotations

from typing import Any

#: Bump this whenever `REQUIRED_METRIC_FIELDS` (or the meaning of an
#: existing field) changes -- old `MetricsResult` rows keep whatever
#: version they were written with, so readers can branch on it instead of
#: assuming every row matches the current shape.
SCHEMA_VERSION = 1

CLASSIFICATION = "classification"
REGRESSION = "regression"
TASK_TYPES = (CLASSIFICATION, REGRESSION)

#: The fields Evaluate can always rely on being present for a given task
#: type, regardless of which feature app (Train/Tune) or which estimator
#: produced the run -- extra estimator-specific fields may ride alongside
#: these, but these are never missing.
REQUIRED_METRIC_FIELDS: dict[str, frozenset[str]] = {
    CLASSIFICATION: frozenset({"accuracy", "precision", "recall", "f1"}),
    REGRESSION: frozenset({"mae", "rmse", "r2"}),
}


class InvalidMetrics(ValueError):
    """Raised when a metrics dict doesn't match its declared task type's
    required shape -- catches a Train/Tune bug at write time instead of
    Evaluate crashing later on a missing key."""


def validate_metrics(task_type: str, metrics: dict[str, Any]) -> None:
    if task_type not in TASK_TYPES:
        raise InvalidMetrics(f"Unknown task_type {task_type!r} (known: {', '.join(TASK_TYPES)})")

    required = REQUIRED_METRIC_FIELDS[task_type]
    missing = required - metrics.keys()
    if missing:
        raise InvalidMetrics(
            f"{task_type} metrics missing required field(s): {sorted(missing)}"
        )
    for field in required:
        if not isinstance(metrics[field], (int, float)):
            raise InvalidMetrics(f"{task_type} metric {field!r} must be numeric, got {type(metrics[field]).__name__}")


def build_classification_metrics(
    *, accuracy: float, precision: float, recall: float, f1: float, **extra: Any
) -> dict[str, Any]:
    """Build (and validate) a classification metrics dict. `extra` carries
    any estimator-specific fields (e.g. `roc_auc`) alongside the required
    ones -- Evaluate reads the required fields generically and may ignore
    or surface `extra` fields per-model."""
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, **extra}
    validate_metrics(CLASSIFICATION, metrics)
    return metrics


def build_regression_metrics(*, mae: float, rmse: float, r2: float, **extra: Any) -> dict[str, Any]:
    metrics = {"mae": mae, "rmse": rmse, "r2": r2, **extra}
    validate_metrics(REGRESSION, metrics)
    return metrics
