"""Phase 1: shared Job/Run model + Celery task wrapper (issue #2).

CELERY_TASK_ALWAYS_EAGER is on by default in this sandbox (see
config/settings.py) so submit_job()'s .delay() call runs synchronously,
in-process, with no Redis broker required -- these tests exercise the exact
same task code that would run on a real worker.
"""
from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model

from automl_core.models import Job
from automl_core.tasks import submit_job

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="alice", password="pw")


@pytest.fixture
def other_user(db):
    return User.objects.create_user(username="bob", password="pw")


def _succeeds(x, y):
    return x + y


def _fails():
    raise RuntimeError("boom")


@pytest.mark.django_db
def test_job_records_status_and_owner(user):
    job = submit_job(
        owner=user,
        feature="train",
        dotted_path="tests.test_jobs._succeeds",
        args=[2, 3],
    )
    job.refresh_from_db()

    assert job.owner_id == user.pk
    assert job.status == Job.Status.SUCCEEDED
    assert job.started_at is not None
    assert job.finished_at is not None


@pytest.mark.django_db
def test_failed_task_marks_job_failed_and_never_silently_drops_the_error(user):
    job = submit_job(owner=user, feature="tune", dotted_path="tests.test_jobs._fails")
    job.refresh_from_db()

    assert job.status == Job.Status.FAILED
    assert "boom" in job.error_message


@pytest.mark.django_db
def test_per_user_job_history_is_queryable(user, other_user):
    submit_job(owner=user, feature="train", dotted_path="tests.test_jobs._succeeds", args=[1, 1])
    submit_job(owner=user, feature="tune", dotted_path="tests.test_jobs._succeeds", args=[1, 1])
    submit_job(owner=other_user, feature="train", dotted_path="tests.test_jobs._succeeds", args=[1, 1])

    assert Job.objects.for_user(user).count() == 2
    assert Job.objects.for_user(other_user).count() == 1
    assert set(Job.objects.for_user(user).values_list("feature", flat=True)) == {"train", "tune"}


@pytest.mark.django_db
def test_submit_job_requires_a_saved_owner():
    with pytest.raises(ValueError):
        submit_job(owner=None, feature="train", dotted_path="tests.test_jobs._succeeds")


@pytest.mark.django_db
def test_a_feature_app_can_create_a_job_without_reimplementing_transitions(user):
    """The whole point of the shared model: one call, no manual status
    bookkeeping in the feature app."""
    job = submit_job(owner=user, feature="evaluate", dotted_path="tests.test_jobs._succeeds", args=[10, 5])

    assert isinstance(job, Job)
    job.refresh_from_db()
    assert job.status in {Job.Status.QUEUED, Job.Status.SUCCEEDED}
