from django.urls import include, path

urlpatterns = [
    path("", include("automl_core.urls")),
]
