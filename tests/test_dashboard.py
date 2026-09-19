"""Dashboard shell tests (issue #4): the feature-picker home view must be
login-gated, list all 3 platform features (Train/Tune/Evaluate) with a
description each, and link the one already-implemented feature (Train) to
its real URL rather than a stub. Train's own flow (issue #5: upload -> pick
target column -> task-type detection) lives at `train:select_target`."""
import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse

User = get_user_model()


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def user(db):
    return User.objects.create_user(username="alice", password="s3cret-pass!")


def test_home_redirects_unauthenticated_users_to_login(client):
    resp = client.get(reverse("dashboard:home"))
    assert resp.status_code == 302
    assert reverse("login") in resp.url


def test_home_lists_every_feature_with_a_description(client, user):
    client.login(username="alice", password="s3cret-pass!")
    resp = client.get(reverse("dashboard:home"))
    assert resp.status_code == 200
    for label in ("Train", "Tune", "Evaluate"):
        assert label.encode() in resp.content
    assert b"Upload a dataset" in resp.content
    assert b"Optuna hyperparameter" in resp.content
    assert b"unified metrics schema" in resp.content


def test_home_links_train_to_its_own_upload_flow(client, user):
    client.login(username="alice", password="s3cret-pass!")
    resp = client.get(reverse("dashboard:home"))
    content = resp.content.decode()
    assert reverse("train:select_target") in content


def test_home_train_is_open_tune_and_evaluate_are_coming_soon(client, user):
    client.login(username="alice", password="s3cret-pass!")
    resp = client.get(reverse("dashboard:home"))
    content = resp.content.decode()
    assert content.count(">Open<") == 1
    assert content.count("Coming soon") == 2


def test_login_page_renders(client):
    resp = client.get(reverse("login"))
    assert resp.status_code == 200
    assert b"Log in" in resp.content


def test_login_then_logout_round_trip(client, user):
    resp = client.post(
        reverse("login"), {"username": "alice", "password": "s3cret-pass!"}
    )
    assert resp.status_code == 302
    resp = client.get(reverse("dashboard:home"))
    assert resp.status_code == 200

    resp = client.post(reverse("logout"))
    resp = client.get(reverse("dashboard:home"))
    assert resp.status_code == 302
