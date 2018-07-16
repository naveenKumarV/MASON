from django import forms
from .models import ImageModel

class ImageUploadForm(forms.ModelForm):
	x = forms.IntegerField(initial=0, min_value=0, required=False)
	y = forms.IntegerField(initial=0, min_value=0, required=False)
	h = forms.IntegerField(initial=0, min_value=0, required=False)
	w = forms.IntegerField(initial=0, min_value=0, required=False)
	class Meta:
		model = ImageModel
		fields = ['image', 'x', 'y', 'h', 'w']

