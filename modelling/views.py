from django.shortcuts import get_object_or_404
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import contrastive_explanation as ce
import itertools

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn import linear_model, naive_bayes

from .models import Dataset
from .serializers import DatasetDetailSerializer, DatasetListSerializer, DatasetPredictSerializer
from .util import fetch_from_openml, train_model, explain_sample, predict_samples


class DatasetViewSet(viewsets.ViewSet):
    @action(detail=False)
    def retrieve_adult(self, request):
        openml_idx = 1590
        columns_to_drop = ['fnlwgt','education-num','capital-loss','capital-gain','workclass', 'education', 'marital-status']

        X, y, _, _, _, _, categorical_names, attribute_names = fetch_from_openml(openml_idx, columns_to_drop=columns_to_drop)
        X['label'] = y
        X = X[(X['occupation'] != 'nan') & (X['native-country'] != 'nan')]

        unique_per_category = {cat: X[cat].unique() for cat in categorical_names}
        bounds_per_feature = {feat: [X[feat].min(), X[feat].max()] for feat in list(set(attribute_names) - set(categorical_names))}

        return Response({"values_per_category": unique_per_category, "bounds_per_feature": bounds_per_feature, "features": attribute_names})

    @action(detail=False)
    def predict_adult(self, request):
        openml_idx = 1590
        columns_to_drop = ['fnlwgt','education-num','capital-loss','capital-gain','workclass', 'education', 'marital-status']

        X, y, train_X, test_X, train_y, test_y, categorical_names, attribute_names = fetch_from_openml(openml_idx, columns_to_drop=columns_to_drop)
        X['label'] = y
        X = X[(X['occupation'] != 'nan') & (X['native-country'] != 'nan')]
        y = X['label']
        X = X.drop(columns=['label'])
        train_X, test_X, train_y, test_y = train_test_split(X, y)

        try:
            model = joblib.load(f'{openml_idx}.sav')
        except:
            model = train_model(naive_bayes.CategoricalNB(), categorical_names, X, train_X, train_y, test_X, test_y)
            joblib.dump(model, f'{openml_idx}.sav')

        df = pd.Series(json.loads(request.GET['sample']))
        prediction = predict_samples(model, np.array(df).reshape(1, -1))

        cf, _, _, _ = explain_sample(df, model, X, [i for i, x in enumerate(attribute_names) if x in categorical_names], None)

        return Response({'explanation': cf, 'prediction': prediction[0]})


    @action(detail=False)
    def nearby_samples(self, request):
        openml_idx = 1590
        columns_to_drop = ['fnlwgt','education-num','capital-loss','capital-gain','workclass', 'education', 'marital-status']

        model = joblib.load(f'{openml_idx}.sav')
        X, _, _, _, _, _, categorical_names, attribute_names = fetch_from_openml(openml_idx, columns_to_drop=columns_to_drop)
        X = X[(X['occupation'] != 'nan') & (X['native-country'] != 'nan')]

        sample = np.array(pd.Series(json.loads(request.GET['sample']))).reshape(1,-1)

        dm = ce.domain_mappers.DomainMapperPandas(X, contrast_names=model[1].classes_, categorical_features=[i for i, x in enumerate(attribute_names) if x in categorical_names])
        cfi = itertools.chain.from_iterable
        categorical_features = list(cfi(dm.feature_map[c]
                                            for c in dm.categorical_features))
        e = LimeTabularExplainer(dm._one_hot_encode(np.array(X)),
                                 categorical_features=categorical_features,
                                 discretize_continuous=False)
        sample = dm.apply_encode(np.array(sample)[0])
        _, neighbor_data = e._LimeTabularExplainer__data_inverse(sample,
                                                                 10)
        predict_data = dm._apply_decode(neighbor_data)

        return Response({'data': predict_data})

    @action(detail=False)
    def update_model(self, request):
        model = joblib.load('1590.sav')

        X = np.array(pd.Series(json.loads(request.GET['sample']))).reshape(1,-1)
        y = pd.Series(request.GET['prediction'])
        X = model.named_steps['label_encoder'].fit_transform(X)

        model.named_steps['classifier'].partial_fit(X, y)
        joblib.dump(model, '1590.sav')

        return Response()
