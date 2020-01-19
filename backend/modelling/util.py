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
    contrast_names = np.array(y.unique())

    for x in attribute_names:
        X[x] = X[x].astype(str)

    train_X, test_X, train_y, test_y = train_test_split(X, y)

    return X, y, train_X, test_X, train_y, test_y, categorical_names, attribute_names, contrast_names


def obtain_data(dataset_index, columns_to_drop=[]):
    return fetch_from_openml(dataset_index, columns_to_drop)


def train_model(classifier, categorical_names, X, train_X, train_y, test_X, test_y):
    model = Pipeline([('label_encoder', ce.CustomLabelEncoder(categorical_names).fit(X)),('classifier', classifier)])
    model.fit(train_X, train_y)
    print('Classifier performance (F1):', f1_score(test_y, model.predict(test_X), average='weighted'))

    return model


def explain_sample(sample, model, contrast_names, X, categorical_features, foil=None):
    dm = ce.domain_mappers.DomainMapperPandas(X, contrast_names=contrast_names, categorical_features=categorical_features)
    tree = ce.TreeExplanator(print_tree=False, domain_mapper=dm, feature_map=dm.feature_map)
    exp = ce.ContrastiveExplanation(dm, tree)

    return exp.explain_instance_domain(model.predict_proba, sample, include_factual=True, foil=foil)
