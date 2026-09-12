"""Celery task wrapper for the shared Job model (Phase 1).

A feature app calls submit_job(...) with a dotted path to whatever function
does its real work; this module handles creating the Job row and driving its
status transitions (queued -> running -> succeeded/failed) so no feature app
has to reimplement that bookkeeping.
"""
from __future__ import annotations

import logging
from typing import Any

from celery import shared_task
from django.utils.module_loading import import_string

from .models import Job

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def run_job(self, job_id: int, dotted_path: str, args: list | None = None, kwargs: dict | None = None) -> Any:
    """Run the callable at ``dotted_path`` and update the Job row accordingly.

    On failure the Job is marked failed with the error message *and* the
    exception is re-raised -- a caught-and-swallowed exception here would
    leave Celery believing the task succeeded while the Job row (the
    user-facing source of truth) says failed, or vice versa if we forgot to
    mark it. Re-raising keeps both in sync and keeps the failure visible in
    Celery's own logging/monitoring too.
    """
    args = args or []
    kwargs = kwargs or {}

    job = Job.objects.get(pk=job_id)
    job.mark_running()
    try:
        func = import_string(dotted_path)
        result = func(*args, **kwargs)
    except Exception as exc:
        logger.exception("Job #%s (%s) failed", job.pk, dotted_path)
        job.mark_failed(str(exc))
        raise
    else:
        job.mark_succeeded()
        return result


def submit_job(*, owner, feature: str, dotted_path: str, args: list | None = None, kwargs: dict | None = None) -> Job:
    """Create a Job row for ``owner`` and enqueue the work behind it.

    Returns immediately with the Job in ``queued`` status (or already
    finished, if CELERY_TASK_ALWAYS_EAGER runs it synchronously before this
    function returns) -- callers should re-fetch the Job to see its current
    status rather than assuming either behavior.
    """
    if owner is None or not getattr(owner, "pk", None):
        raise ValueError("submit_job requires a saved user instance as `owner`")

    job = Job.objects.create(owner=owner, feature=feature)
    run_job.delay(job.id, dotted_path, args, kwargs)
    return job
