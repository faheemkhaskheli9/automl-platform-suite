from django.apps import AppConfig


class TrainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "train"
    verbose_name = "Train (dataset -> target selection -> multi-model training)"
