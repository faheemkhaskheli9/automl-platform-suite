import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse


@pytest.fixture
def client():
    return Client()


def test_get_upload_page_renders_form(client):
    resp = client.get(reverse("automl_core:upload_dataset"))
    assert resp.status_code == 200
    assert b"Upload a dataset" in resp.content


def test_post_valid_csv_shows_preview(client):
    upload = SimpleUploadedFile(
        "data.csv", b"a,b\n1,2\n3,4\n", content_type="text/csv"
    )
    resp = client.post(
        reverse("automl_core:upload_dataset"), {"dataset_file": upload}
    )
    assert resp.status_code == 200
    assert b"Preview: data.csv" in resp.content
    assert b"2 rows" in resp.content


def test_post_invalid_file_shows_error_not_preview(client):
    upload = SimpleUploadedFile(
        "data.txt", b"not tabular", content_type="text/plain"
    )
    resp = client.post(
        reverse("automl_core:upload_dataset"), {"dataset_file": upload}
    )
    assert resp.status_code == 200
    assert b"Error:" in resp.content
    assert b"Unsupported file type" in resp.content
    assert b"Preview:" not in resp.content


def test_post_oversized_file_is_rejected(client, settings):
    settings.AUTOML_MAX_UPLOAD_BYTES = 5
    upload = SimpleUploadedFile(
        "data.csv", b"a,b\n1,2\n3,4\n", content_type="text/csv"
    )
    resp = client.post(
        reverse("automl_core:upload_dataset"), {"dataset_file": upload}
    )
    assert resp.status_code == 200
    assert b"too large" in resp.content
