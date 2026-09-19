from django.urls import path

from . import views

app_name = "train"

urlpatterns = [
    path("train/select-target/", views.select_target, name="select_target"),
]
