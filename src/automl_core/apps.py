from django.apps import AppConfig


class AutomlCoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "automl_core"
    verbose_name = "AutoML Core (shared upload/validation/job/metrics layer)"
