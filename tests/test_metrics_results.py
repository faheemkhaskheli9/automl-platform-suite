"""Tests for issue #3: MetricsResult.record() is the one way any feature
app writes metrics, and Evaluate can read any run's metrics generically.

Train and Tune feature apps land in later phases (#7, #10) -- these tests
stand in for them by calling MetricsResult.record() the same way those
apps are expected to, proving the shared schema/model already works for
both without per-feature special-casing.
"""
from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model

from automl_core.metrics_schema import (
    SCHEMA_VERSION,
    InvalidMetrics,
    build_classification_metrics,
    build_regression_metrics,
)
from automl_core.models import Job, MetricsResult

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="alice", password="pw")


@pytest.fixture
def job_factory(user):
    def make(feature: str) -> Job:
        return Job.objects.create(owner=user, feature=feature)

    return make


@pytest.mark.django_db
def test_train_feature_app_writes_classification_metrics_in_the_shared_schema(job_factory):
    job = job_factory("train")
    metrics = build_classification_metrics(accuracy=0.92, precision=0.9, recall=0.88, f1=0.89)

    result = MetricsResult.record(job, "classification", metrics)

    assert result.job_id == job.pk
    assert result.schema_version == SCHEMA_VERSION
    assert result.task_type == "classification"
    assert result.metrics == metrics


@pytest.mark.django_db
def test_tune_feature_app_writes_regression_metrics_in_the_shared_schema(job_factory):
    job = job_factory("tune")
    metrics = build_regression_metrics(mae=0.5, rmse=0.7, r2=0.6)

    result = MetricsResult.record(job, "regression", metrics)

    assert result.schema_version == SCHEMA_VERSION
    assert result.task_type == "regression"
    assert result.metrics == metrics


@pytest.mark.django_db
def test_record_rejects_metrics_that_dont_match_the_declared_task_type(job_factory):
    job = job_factory("train")
    with pytest.raises(InvalidMetrics):
        MetricsResult.record(job, "classification", {"mae": 1.0, "rmse": 1.0, "r2": 0.5})


@pytest.mark.django_db
def test_evaluate_can_read_any_runs_metrics_without_per_feature_special_casing(job_factory):
    train_job = job_factory("train")
    tune_job = job_factory("tune")
    MetricsResult.record(train_job, "classification", build_classification_metrics(
        accuracy=0.9, precision=0.9, recall=0.9, f1=0.9
    ))
    MetricsResult.record(tune_job, "classification", build_classification_metrics(
        accuracy=0.95, precision=0.95, recall=0.95, f1=0.95
    ))

    # "Evaluate" here is just: read every MetricsResult, generically, with
    # no branch on job.feature.
    all_results = MetricsResult.objects.select_related("job").all()
    accuracies = {r.job.feature: r.metrics["accuracy"] for r in all_results}

    assert accuracies == {"train": 0.9, "tune": 0.95}


@pytest.mark.django_db
def test_a_job_can_have_at_most_one_metrics_result_per_write_but_history_is_kept(job_factory):
    job = job_factory("train")
    MetricsResult.record(job, "classification", build_classification_metrics(
        accuracy=0.8, precision=0.8, recall=0.8, f1=0.8
    ))
    MetricsResult.record(job, "classification", build_classification_metrics(
        accuracy=0.85, precision=0.85, recall=0.85, f1=0.85
    ))

    assert job.metrics_results.count() == 2
    latest = job.metrics_results.first()  # default ordering: -created_at
    assert latest.metrics["accuracy"] == 0.85
