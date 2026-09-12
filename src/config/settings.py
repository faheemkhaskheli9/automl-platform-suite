"""
Django settings for the AutoML Platform Suite skeleton (Phase 1).

Only what `automl_core`'s dataset upload/validation/preview flow needs is
wired up here; feature apps (`train`, `tune`, `evaluate`) and Celery/MLflow
will be layered on in later phases per README.md Section 5.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "insecure-dev-key-change-me")

DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "automl_core",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# DATABASE_URL is not parsed here (no Postgres driver needed for Phase 1);
# local/dev always uses SQLite per README.md Section 3.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

STATIC_URL = "static/"

# --- automl_core: dataset upload settings -----------------------------
# Max upload size (bytes) before validation even opens the file.
AUTOML_MAX_UPLOAD_BYTES = int(os.environ.get("AUTOML_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
# How many rows to show in the post-upload preview table.
AUTOML_PREVIEW_ROWS = int(os.environ.get("AUTOML_PREVIEW_ROWS", 10))
