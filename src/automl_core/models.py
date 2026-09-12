"""Shared Job/Run model (Phase 1): every feature app (train/tune/evaluate)
records its background work the same way instead of inventing its own job
tracking -- see automl_core/tasks.py for the Celery wrapper that drives the
status transitions below."""
from __future__ import annotations

from django.conf import settings
from django.db import models
from django.utils import timezone

from .metrics_schema import SCHEMA_VERSION, TASK_TYPES, validate_metrics


class JobQuerySet(models.QuerySet):
    def for_user(self, user) -> "JobQuerySet":
        return self.filter(owner=user)


class Job(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="automl_jobs",
    )
    # Which feature app submitted this job -- "train", "tune", "evaluate" in
    # later phases. Not an enum/FK yet since only automl_core exists so far;
    # a feature app is free to use any short slug here.
    feature = models.CharField(max_length=32)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.QUEUED
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    # Populated only on failure -- never silently dropped, see tasks.run_job.
    error_message = models.TextField(blank=True, default="")

    objects = JobQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover - repr only
        return f"Job #{self.pk} ({self.feature}, {self.status})"

    def mark_running(self) -> None:
        self.status = self.Status.RUNNING
        self.started_at = timezone.now()
        self.save(update_fields=["status", "started_at"])

    def mark_succeeded(self) -> None:
        self.status = self.Status.SUCCEEDED
        self.finished_at = timezone.now()
        self.save(update_fields=["status", "finished_at"])

    def mark_failed(self, error_message: str) -> None:
        self.status = self.Status.FAILED
        self.finished_at = timezone.now()
        self.error_message = error_message
        self.save(update_fields=["status", "finished_at", "error_message"])


class MetricsResult(models.Model):
    """One run's metrics, in the schema every feature app shares (issue
    #3). Evaluate reads these generically off `task_type`/`metrics` with
    no per-feature (train vs. tune) special-casing -- `job.feature` is
    only ever used for display/filtering, never to pick a different
    metrics shape.

    `schema_version` is stamped from `metrics_schema.SCHEMA_VERSION` at
    write time (never recomputed later), so a future schema change can't
    silently reinterpret an already-stored run under the new shape.
    """

    TASK_TYPE_CHOICES = [(t, t.capitalize()) for t in TASK_TYPES]

    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="metrics_results")
    schema_version = models.PositiveIntegerField()
    task_type = models.CharField(max_length=16, choices=TASK_TYPE_CHOICES)
    metrics = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover - repr only
        return f"MetricsResult(job={self.job_id}, task_type={self.task_type})"

    @classmethod
    def record(cls, job: Job, task_type: str, metrics: dict) -> "MetricsResult":
        """The one way any feature app (Train, Tune, ...) writes metrics --
        validates against the shared schema before persisting, so a
        malformed write fails loudly at write time instead of Evaluate
        hitting a missing/wrong-typed field later."""
        validate_metrics(task_type, metrics)
        return cls.objects.create(
            job=job, schema_version=SCHEMA_VERSION, task_type=task_type, metrics=metrics
        )
