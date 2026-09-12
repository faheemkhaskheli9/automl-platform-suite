"""Shared Job/Run model (Phase 1): every feature app (train/tune/evaluate)
records its background work the same way instead of inventing its own job
tracking -- see automl_core/tasks.py for the Celery wrapper that drives the
status transitions below."""
from __future__ import annotations

from django.conf import settings
from django.db import models
from django.utils import timezone


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
