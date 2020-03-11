from django.shortcuts import get_object_or_404
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn import linear_model

from .models import Dataset
from .serializers import DatasetDetailSerializer, DatasetListSerializer, DatasetPredictSerializer
from .util import obtain_data, train_model, explain_sample, predict_samples

import time


class DatasetViewSet(viewsets.ViewSet):
    @staticmethod
    def list(request):
        queryset = Dataset.objects.all()
        serializer = DatasetListSerializer(queryset, many=True)
        return Response(serializer.data)

    @staticmethod
    def retrieve(request, pk=None):
        queryset = Dataset.objects.all()
        dataset = get_object_or_404(queryset, openml_idx=pk)
        serializer = DatasetDetailSerializer(dataset)
        return Response(serializer.data)

    @action(detail=True)
    def retrieve_samples(self, request, pk=None):
        queryset = Dataset.objects.all()
        dataset = get_object_or_404(queryset, openml_idx=pk)
        serializer = DatasetPredictSerializer(dataset)

        openml_idx = serializer.data['openml_idx']
        columns_to_drop = serializer.data['columns_to_drop'].split(',')
        if columns_to_drop[0] == '':
            columns_to_drop = []

        X, y, train_X, test_X, train_y, test_y, categorical_names, _ = obtain_data(openml_idx, columns_to_drop=columns_to_drop)

        try:
            model = joblib.load(f'{openml_idx}.sav')
        except:
            model = train_model(AdaBoostClassifier(random_state=np.random.RandomState(1994), n_estimators=1000), categorical_names, X, train_X, train_y, test_X, test_y)
            joblib.dump(model, f'{openml_idx}.sav')

        X['model_predictions'] = predict_samples(model, X)
        X['label'] = y

        unique_per_category = {cat: X[cat].unique() for cat in categorical_names}
        unique_per_category['label'] = X['label'].unique()

        return Response({"categorical_values": unique_per_category, "samples": X.to_dict('records')})


    @action(detail=True)
    def predict_sample(self, request, pk=None):
        start = time.time()

        queryset = Dataset.objects.all()
        dataset = get_object_or_404(queryset, openml_idx=pk)
        serializer = DatasetPredictSerializer(dataset)

        openml_idx = serializer.data['openml_idx']
        columns_to_drop = serializer.data['columns_to_drop'].split(',')
        if columns_to_drop[0] == '':
            columns_to_drop = []

        X, y, train_X, test_X, train_y, test_y, categorical_names, attribute_names = obtain_data(openml_idx, columns_to_drop=columns_to_drop)

        dataset_loading_time = time.time()
        print(f'Loading data: {dataset_loading_time-start}s')

        try:
            model = joblib.load(f'{openml_idx}.sav')
        except:
            model = train_model(AdaBoostClassifier(random_state=np.random.RandomState(1994), n_estimators=1000), categorical_names, X, train_X, train_y, test_X, test_y)
            joblib.dump(model, f'{openml_idx}.sav')

        model_loading_time = time.time()
        print(f'Loading model: {model_loading_time - dataset_loading_time}s')

        sample_idx = int(request.GET['sampleId'])
        if (sample_idx == -1 or sample_idx > len(X)):
            sample_idx = np.random.randint(len(X))
        sample = X.iloc[sample_idx]

        foil = request.GET['foilClass'] if 'foilClass' in request.GET else None
        cf, f, cf_rules, f_rules = explain_sample(sample, model, X, [i for i, x in enumerate(attribute_names) if x in categorical_names], foil)

        print(f'Explanation: {time.time() - model_loading_time}s')
        return Response(cf)

    @action(detail=True)
    def retrieve_spam(self, request, pk=None):
        queryset = Dataset.objects.all()
        dataset = get_object_or_404(queryset, openml_idx=pk)
        serializer = DatasetPredictSerializer(dataset)

        openml_idx = serializer.data['openml_idx']

        X, y, train_X, test_X, train_y, test_y, categorical_names, _ = obtain_data(openml_idx)
        X['label'] = y

        unique_per_category = {cat: X[cat].unique() for cat in categorical_names}
        unique_per_category['label'] = X['label'].unique()

        return Response({"categorical_values": unique_per_category, "samples": X.to_dict('records')})

    @action(detail=True)
    def predict_spam(self, request, pk=None):
        start = time.time()

        queryset = Dataset.objects.all()
        dataset = get_object_or_404(queryset, openml_idx=pk)
        serializer = DatasetPredictSerializer(dataset)

        openml_idx = serializer.data['openml_idx']
        columns_to_drop = serializer.data['columns_to_drop'].split(',')
        if columns_to_drop[0] == '':
            columns_to_drop = []

        X, y, train_X, test_X, train_y, test_y, categorical_names, attribute_names = obtain_data(openml_idx, columns_to_drop=columns_to_drop)

        dataset_loading_time = time.time()
        print(f'Loading data: {dataset_loading_time-start}s')

        try:
            model = joblib.load(f'{openml_idx}.sav')
        except:
            model = train_model(linear_model.SGDClassifier(loss="log"), categorical_names, X, train_X, train_y, test_X, test_y)
            joblib.dump(model, f'{openml_idx}.sav')

        model_loading_time = time.time()
        print(f'Loading model: {model_loading_time - dataset_loading_time}s')
        df = pd.Series(json.loads(request.GET['sample']))
        cf, f, cf_rules, f_rules = explain_sample(df, model, X, [i for i, x in enumerate(attribute_names) if x in categorical_names], None)

        print(f'Explanation: {time.time() - model_loading_time}s')
        prediction = predict_samples(model, np.array(df).reshape(1, -1))
        return Response({'explanation': cf, 'prediction': prediction[0]})

    @action(detail=True)
    def update_model(self, request, pk=None):
        start = time.time()

        model = joblib.load(f'{pk}.sav')

        model_loading_time = time.time()
        print(f'Loading model: {model_loading_time - start}s')

        X = np.array(pd.Series(json.loads(request.GET['sample']))).reshape(1,-1)
        y = pd.Series(request.GET['prediction'])

        model.named_steps['classifier'].partial_fit(X, y)
        joblib.dump(model, f'{pk}.sav')
        return Response()

    @action(detail=True)
    def retrieve_adult(self, request, pk=None):
        queryset = Dataset.objects.all()
        dataset = get_object_or_404(queryset, openml_idx=pk)
        serializer = DatasetPredictSerializer(dataset)

        openml_idx = serializer.data['openml_idx']
        columns_to_drop = serializer.data['columns_to_drop'].split(',')
        if columns_to_drop[0] == '':
            columns_to_drop = []

        X, y, train_X, test_X, train_y, test_y, categorical_names, attribute_names = obtain_data(openml_idx, columns_to_drop=columns_to_drop)
        X['label'] = y
        X = X[(X['workclass'] != 'nan') & (X['occupation'] != 'nan') & (X['native-country'] != 'nan')]
        unique_per_category = {cat: X[cat].unique() for cat in categorical_names}
        bounds_per_feature = {feat: [X[feat].min(), X[feat].max()] for feat in list(set(attribute_names) - set(categorical_names))}
        return Response({"values_per_category": unique_per_category, "bounds_per_feature": bounds_per_feature, "features": attribute_names})
