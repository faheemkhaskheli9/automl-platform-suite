"""Celery application for background training/tuning jobs (Phase 1).

CELERY_TASK_ALWAYS_EAGER (see settings.py) defaults on in this sandbox --
there is no Redis broker running here -- so `.delay()` executes the task
synchronously in the calling process instead of being picked up by a worker.
The task code itself (automl_core/tasks.py) is unchanged either way; only
where it runs differs.
"""
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("automl_platform_suite")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
