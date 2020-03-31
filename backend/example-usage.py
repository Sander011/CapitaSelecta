import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import linear_model, naive_bayes
import json

from modelling.util import obtain_data, train_model, explain_sample, predict_samples

# Uninteresting columns
columns_to_drop = ['fnlwgt','education-num','capital-loss','capital-gain','workclass','education','marital-status']

# Index of sample to explain
sample_idx = 0

# Retrieve data and filter incorrect values
X, y, train_X, test_X, train_y, test_y, categorical_names, attribute_names = obtain_data(1590, columns_to_drop=columns_to_drop)
X['label'] = y
X = X[(X['occupation'] != 'nan') & (X['native-country'] != 'nan')]
y = X['label']
X = X.drop(columns=['label'])
train_X, test_X, train_y, test_y = train_test_split(X, y)

# Load black box model (or train one)
try:
    model = joblib.load('1590.sav')
except:
    model = train_model(naive_bayes.CategoricalNB(), categorical_names, X, train_X, train_y, test_X, test_y)
    joblib.dump(model, '1590.sav')

# Locate sample
df = X.iloc[sample_idx]

# Predict
prediction = predict_samples(model, np.array(df).reshape(1, -1))

# Obtain rules
cf, f, cf_rules, f_rules = explain_sample(df, model, X, [i for i, x in enumerate(attribute_names) if x in categorical_names], None, print_tree=True)
print(f'Counterfactual: {cf}')
print(f'Factual: {f}')
print(f'Prediction: {prediction}')