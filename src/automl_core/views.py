from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .forms import DatasetUploadForm
from .validation import DatasetValidationError, validate_dataset


@require_http_methods(["GET", "POST"])
def upload_dataset(request):
    """Upload a tabular dataset, validate it, and show a preview.

    Acceptance criteria (issue #1): CSV + one other tabular format accepted,
    schema/type validated before acceptance, a preview table is shown on
    success, and an invalid file is rejected with a clear error rather than a
    silent partial import (the file is never persisted/cached until it has
    fully passed `validate_dataset`).
    """
    preview = None
    error = None

    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.cleaned_data["dataset_file"]
            max_bytes = settings.AUTOML_MAX_UPLOAD_BYTES
            if upload.size > max_bytes:
                error = (
                    f"File is too large ({upload.size} bytes); "
                    f"max is {max_bytes} bytes."
                )
            else:
                try:
                    preview = validate_dataset(
                        upload.file,
                        upload.name,
                        preview_rows=settings.AUTOML_PREVIEW_ROWS,
                    )
                except DatasetValidationError as exc:
                    error = exc.reason
        else:
            error = "No file was submitted."
    else:
        form = DatasetUploadForm()

    return render(
        request,
        "automl_core/upload.html",
        {"form": form, "preview": preview, "error": error},
    )
