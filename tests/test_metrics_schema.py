"""Framework-free tests for issue #3's metrics schema (no Django/db)."""

from __future__ import annotations

import pytest

from automl_core.metrics_schema import (
    CLASSIFICATION,
    REGRESSION,
    SCHEMA_VERSION,
    InvalidMetrics,
    build_classification_metrics,
    build_regression_metrics,
    validate_metrics,
)


def test_schema_version_is_a_stable_int():
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_build_classification_metrics_returns_required_fields():
    metrics = build_classification_metrics(accuracy=0.9, precision=0.8, recall=0.85, f1=0.82)
    assert metrics == {"accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1": 0.82}


def test_build_regression_metrics_returns_required_fields():
    metrics = build_regression_metrics(mae=1.2, rmse=1.5, r2=0.7)
    assert metrics == {"mae": 1.2, "rmse": 1.5, "r2": 0.7}


def test_classification_metrics_can_carry_extra_estimator_specific_fields():
    metrics = build_classification_metrics(accuracy=0.9, precision=0.8, recall=0.85, f1=0.82, roc_auc=0.95)
    assert metrics["roc_auc"] == 0.95


def test_validate_metrics_rejects_unknown_task_type():
    with pytest.raises(InvalidMetrics):
        validate_metrics("clustering", {})


def test_validate_metrics_rejects_missing_required_field():
    with pytest.raises(InvalidMetrics, match="missing required field"):
        validate_metrics(CLASSIFICATION, {"accuracy": 0.9, "precision": 0.8, "recall": 0.85})


def test_validate_metrics_rejects_non_numeric_value():
    with pytest.raises(InvalidMetrics, match="must be numeric"):
        validate_metrics(REGRESSION, {"mae": "bad", "rmse": 1.0, "r2": 0.5})
