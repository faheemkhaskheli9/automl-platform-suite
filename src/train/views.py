"""Train feature views (issue #5): upload a dataset, then pick a target
column and see its inferred task type, with an explicit override.

This is a two-step flow served by one view: the first POST (the plain
dataset-upload form) validates and previews the dataset; the second POST
(`TargetSelectionForm`) also carries the validated bytes back as hidden
fields, so this step doesn't need a Dataset model (not built yet) or
session storage to bridge the two requests.
"""
from __future__ import annotations

import base64
import binascii

from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from automl_core.forms import DatasetUploadForm
from automl_core.validation import (
    DatasetValidationError,
    load_dataframe_bytes,
    validate_dataset_bytes,
)

from .forms import TASK_TYPE_CHOICES, TargetSelectionForm
from .task_type import TargetColumnError, analyze_target_column


@require_http_methods(["GET", "POST"])
def select_target(request):
    upload_form = DatasetUploadForm()
    preview = None
    analysis = None
    error = None
    dataset_b64 = ""
    selected_column = ""
    selected_override = ""

    if request.method == "POST" and request.FILES.get("dataset_file"):
        # Step 1: validate the upload and prepare the round-trip payload.
        upload_form = DatasetUploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            upload = upload_form.cleaned_data["dataset_file"]
            max_bytes = settings.AUTOML_MAX_UPLOAD_BYTES
            if upload.size > max_bytes:
                error = (
                    f"File is too large ({upload.size} bytes); "
                    f"max is {max_bytes} bytes."
                )
            else:
                content = upload.read()
                try:
                    preview = validate_dataset_bytes(
                        content, upload.name, preview_rows=settings.AUTOML_PREVIEW_ROWS
                    )
                except DatasetValidationError as exc:
                    error = exc.reason
                else:
                    dataset_b64 = base64.b64encode(content).decode("ascii")
        else:
            error = "No file was submitted."

    elif request.method == "POST" and "target_column" in request.POST:
        # Step 2: analyze the chosen target column against the same bytes.
        target_form = TargetSelectionForm(request.POST)
        if target_form.is_valid():
            raw_b64 = target_form.cleaned_data["dataset_b64"]
            filename = target_form.cleaned_data["dataset_filename"]
            selected_column = target_form.cleaned_data["target_column"]
            selected_override = target_form.cleaned_data["task_type_override"] or ""

            try:
                content = base64.b64decode(raw_b64, validate=True)
            except (binascii.Error, ValueError):
                error = "The dataset could not be re-read; please re-upload it."
                content = None

            if content is not None:
                max_bytes = settings.AUTOML_MAX_UPLOAD_BYTES
                if len(content) > max_bytes:
                    error = (
                        f"File is too large ({len(content)} bytes); "
                        f"max is {max_bytes} bytes."
                    )
                else:
                    dataset_b64 = raw_b64
                    try:
                        preview = validate_dataset_bytes(
                            content, filename, preview_rows=settings.AUTOML_PREVIEW_ROWS
                        )
                        df = load_dataframe_bytes(content, filename)
                    except DatasetValidationError as exc:
                        error = exc.reason
                    else:
                        try:
                            analysis = analyze_target_column(
                                df,
                                selected_column,
                                override_task_type=selected_override or None,
                            )
                        except TargetColumnError as exc:
                            error = str(exc)
        else:
            error = "Invalid submission; please re-upload the dataset."

    return render(
        request,
        "train/select_target.html",
        {
            "upload_form": upload_form,
            "preview": preview,
            "analysis": analysis,
            "error": error,
            "dataset_b64": dataset_b64,
            "selected_column": selected_column,
            "selected_override": selected_override,
            "task_type_choices": TASK_TYPE_CHOICES,
        },
    )
