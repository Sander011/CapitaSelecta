from django.contrib import admin
from .models import Dataset


class DatasetAdmin(admin.ModelAdmin):
    list_display = ('title', 'categorical_names', 'attribute_names', 'contrast_names')


admin.site.register(Dataset, DatasetAdmin)
