from django import forms


class DatasetUploadForm(forms.Form):
    dataset_file = forms.FileField(required=True)
