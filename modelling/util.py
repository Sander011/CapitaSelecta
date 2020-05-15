import numpy as np

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import contrastive_explanation as ce

import openml


def fetch_from_openml(dataset_index, columns_to_drop):
    dataset = openml.datasets.get_dataset(dataset_index)

    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format='dataframe',
                                                                    target=dataset.default_target_attribute)
    X = X.drop(columns_to_drop, axis=1)
    categorical_names = [attribute_names[i] for i in range(len(attribute_names)) if categorical_indicator[i]]
    attribute_names = X.columns

    for x in attribute_names:
        X[x] = X[x].astype(str)

    train_X, test_X, train_y, test_y = train_test_split(X, y)

    return X, y, train_X, test_X, train_y, test_y, [x for x in categorical_names if x not in columns_to_drop], attribute_names


def train_model(classifier, categorical_names, X, train_X, train_y, test_X, test_y):
    model = Pipeline([('label_encoder', ce.CustomLabelEncoder(categorical_names).fit(X)),('classifier', classifier)])
    model.fit(train_X, train_y)
    print('Classifier performance (F1):', f1_score(test_y, model.predict(test_X), average='weighted'))

    return model


def explain_sample(sample, model, X, categorical_features, foil=None):
    dm = ce.domain_mappers.DomainMapperPandas(X, contrast_names=model[1].classes_, categorical_features=categorical_features)
    exp = ce.ContrastiveExplanation(dm)

    return exp.explain_instance_domain(model.predict_proba, sample, foil=foil)


def predict_samples(model, X):
    return [model[1].classes_[np.argmax(x)] for x in model.predict_proba(X)]