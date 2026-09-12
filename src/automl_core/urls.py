from django.urls import path

from . import views

app_name = "automl_core"

urlpatterns = [
    path("upload/", views.upload_dataset, name="upload_dataset"),
]
