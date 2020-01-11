from rest_framework import serializers
from .models import Dataset


class DatasetDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ('id', 'title', 'categorical_names', 'attribute_names', 'contrast_names')


class DatasetListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ('id', 'title')


class DatasetPredictSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ('id', 'openml_idx', 'columns_to_drop')
