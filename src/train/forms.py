from django import forms

from automl_core.metrics_schema import TASK_TYPES

TASK_TYPE_CHOICES = [(t, t.capitalize()) for t in TASK_TYPES]


class TargetSelectionForm(forms.Form):
    """Step 2 of the target-selection flow (issue #5).

    `automl_core` has no Dataset persistence yet (only Job/MetricsResult),
    so the validated upload is round-tripped through this step as hidden
    fields (base64 bytes + filename) rather than re-fetched from storage --
    adding a Dataset model is out of this issue's scope.
    """

    dataset_b64 = forms.CharField(widget=forms.HiddenInput())
    dataset_filename = forms.CharField(widget=forms.HiddenInput())
    target_column = forms.CharField()
    task_type_override = forms.ChoiceField(
        choices=[("", "Auto-detect")] + TASK_TYPE_CHOICES, required=False
    )
