from django import forms


class Folder_name_form(forms.Form):
    project_name = forms.CharField(max_length=500)
    images_to_capture = forms.IntegerField(max_value=1000)
    object_1_label = forms.CharField(max_length=500)
    object_2_label = forms.CharField(max_length=500)
    data_augmentation = forms.BooleanField(required=False)
    no_of_epochs = forms.IntegerField(max_value=5)
    collection_interval = forms.BooleanField(required=False)
