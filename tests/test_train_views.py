"""Train app view tests (issue #5): upload -> preview -> pick a target
column -> see the inferred/overridden task type, all through the public
HTTP flow (mirrors tests/test_views.py's pattern for automl_core)."""
import base64

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse


@pytest.fixture
def client():
    return Client()


def _upload(client, content: bytes, filename: str = "data.csv"):
    upload = SimpleUploadedFile(filename, content, content_type="text/csv")
    return client.post(reverse("train:select_target"), {"dataset_file": upload})


def test_get_page_renders_upload_form(client):
    resp = client.get(reverse("train:select_target"))
    assert resp.status_code == 200
    assert b"Train: select a target column" in resp.content


def test_step1_upload_shows_preview_and_column_picker(client):
    content = b"label,x\n0,1\n1,2\n0,3\n1,4\n"
    resp = _upload(client, content)
    assert resp.status_code == 200
    assert b"Preview: data.csv" in resp.content
    assert b'name="target_column"' in resp.content
    assert b'value="label"' in resp.content


def test_step2_selecting_target_infers_classification(client):
    content = b"label,x\n0,1\n1,2\n0,3\n1,4\n0,5\n1,6\n"
    step1 = _upload(client, content)
    dataset_b64 = base64.b64encode(content).decode("ascii")
    assert dataset_b64.encode() in step1.content

    resp = client.post(
        reverse("train:select_target"),
        {
            "dataset_b64": dataset_b64,
            "dataset_filename": "data.csv",
            "target_column": "label",
            "task_type_override": "",
        },
    )
    assert resp.status_code == 200
    assert b"Detected task type: classification" in resp.content
    assert b"Effective task type: classification" in resp.content


def test_step2_override_changes_effective_task_type(client):
    content = b"label,x\n0,1\n1,2\n0,3\n1,4\n0,5\n1,6\n"
    step1 = _upload(client, content)
    dataset_b64 = base64.b64encode(content).decode("ascii")
    assert dataset_b64.encode() in step1.content

    resp = client.post(
        reverse("train:select_target"),
        {
            "dataset_b64": dataset_b64,
            "dataset_filename": "data.csv",
            "target_column": "label",
            "task_type_override": "regression",
        },
    )
    assert resp.status_code == 200
    assert b"Effective task type: regression" in resp.content
    assert b"(overridden by user)" in resp.content


def test_step2_invalid_target_column_shows_clear_error_not_crash(client):
    content = b"label,x\n0,1\n1,2\n0,3\n1,4\n"
    _upload(client, content)
    dataset_b64 = base64.b64encode(content).decode("ascii")

    resp = client.post(
        reverse("train:select_target"),
        {
            "dataset_b64": dataset_b64,
            "dataset_filename": "data.csv",
            "target_column": "does_not_exist",
            "task_type_override": "",
        },
    )
    assert resp.status_code == 200
    assert b"Error:" in resp.content
    assert b"not found" in resp.content


def test_step2_degenerate_target_column_shows_clear_error_not_crash(client):
    content = b"label,x\n1,1\n1,2\n1,3\n1,4\n"
    _upload(client, content)
    dataset_b64 = base64.b64encode(content).decode("ascii")

    resp = client.post(
        reverse("train:select_target"),
        {
            "dataset_b64": dataset_b64,
            "dataset_filename": "data.csv",
            "target_column": "label",
            "task_type_override": "",
        },
    )
    assert resp.status_code == 200
    assert b"Error:" in resp.content
    assert b"degenerate" in resp.content
