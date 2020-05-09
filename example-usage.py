import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import linear_model, naive_bayes, datasets, ensemble, svm, neural_network, metrics
import json
import contrastive_explanation as ce
import openml

SEED = np.random.RandomState(2020)

iris = datasets.load_iris()
diabetes = datasets.load_diabetes()
heart = pd.read_csv('./heart.csv')

def perform_tests(dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=SEED)

    random_forest = ensemble.RandomForestClassifier(random_state=SEED, n_estimators=500)
    logistic_regression = linear_model.LogisticRegression(random_state=SEED)
    SVM = svm.SVC(random_state=SEED, probability=True)
    nn = neural_network.MLPClassifier()

    models = [random_forest, logistic_regression, SVM, nn]
    model_to_str = ['Random Forest', 'Logistic Regression', 'SVM', 'Neural Network']
    count = 0
    for model in models:
        model.fit(x_train, y_train)
        dm = ce.domain_mappers.DomainMapperTabular(x_train, feature_names=dataset.feature_names, contrast_names=dataset.target_names)
        exp = ce.ContrastiveExplanation(dm)
        times = np.array([])
        lengths = np.array([])
        times_wrong = 0
        model_wrong = 0
        for i in range(len(x_test)):
            sample = x_test[i]
            label = y_test[i]
            start = timer()
            _, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            if len(counterfactuals) == 0:
                times_wrong += 1
            times = np.append(times, end - start)
            lengths = np.append(lengths, len(counterfactuals))
            if model.predict(sample.reshape(1,-1))[0] != label:
                model_wrong += 1
        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.f1_score(y_test, model.predict(x_test), average="weighted")}')
        print(f'Mean length: {np.average(lengths)}')
        print(f'Accuracy: {1 - (times_wrong / len(x_test))}')
        print(f'Fidelity: {1 - (model_wrong / len(x_test))}')
        print(f'Time: {np.average(times)}')

        count += 1


def perform_tests_regression(dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=SEED)

    random_forest = ensemble.RandomForestRegressor(random_state=SEED, n_estimators=500)
    logistic_regression = linear_model.LogisticRegression(random_state=SEED)
    SVM = svm.SVC(random_state=SEED, probability=True)
    nn = neural_network.MLPClassifier()

    models = [random_forest, logistic_regression, SVM, nn]
    model_to_str = ['Random Forest', 'Logistic Regression', 'SVM', 'Neural Network']
    count = 0

    for model in models:
        model.fit(x_train, y_train)
        dm = ce.domain_mappers.DomainMapperTabular(x_train, feature_names=dataset.feature_names)
        exp = ce.ContrastiveExplanation(dm, regression=True)
        times = np.array([])
        lengths = np.array([])
        times_wrong = 0
        model_wrong = 0

        for i in range(len(x_test)):
            sample = x_test[i]
            label = y_test[i]
            start = timer()
            _, counterfactuals = exp.explain_instance_domain(model.predict, sample)
            end = timer()
            if not counterfactuals or len(counterfactuals) == 0:
                times_wrong += 1
            times = np.append(times, end - start)
            lengths = np.append(lengths, 0 if not counterfactuals else len(counterfactuals))
            if model.predict(sample.reshape(1,-1))[0] != label:
                model_wrong += 1

        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.r2_score(y_test, model.predict(x_test))}')
        print(f'Mean length: {np.average(lengths)}')
        print(f'Accuracy: {1 - (times_wrong / len(x_test))}')
        print(f'Fidelity: {1 - (model_wrong / len(x_test))}')
        print(f'Time: {np.average(times)}')

        count += 1


def perform_tests_pandas(dataset):
    target = dataset['target']
    data = dataset.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=SEED)

    random_forest = ensemble.RandomForestClassifier(random_state=SEED, n_estimators=500)
    logistic_regression = linear_model.LogisticRegression(random_state=SEED)
    SVM = svm.SVC(random_state=SEED, probability=True)
    nn = neural_network.MLPClassifier()

    models = [random_forest, logistic_regression, SVM, nn]
    model_to_str = ['Random Forest', 'Logistic Regression', 'SVM', 'Neural Network']
    count = 0
    for model in models:
        model.fit(x_train, y_train)
        dm = ce.domain_mappers.DomainMapperPandas(x_train)
        exp = ce.ContrastiveExplanation(dm)
        times = np.array([])
        lengths = np.array([])
        times_wrong = 0
        model_wrong = 0
        for i in range(len(x_test)):
            sample = x_test.iloc[i]
            label = y_test.iloc[i]
            start = timer()
            _, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            if len(counterfactuals) == 0:
                times_wrong += 1
            times = np.append(times, end - start)
            lengths = np.append(lengths, len(counterfactuals))
            if model.predict([sample])[0] != label:
                model_wrong += 1
        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.f1_score(y_test, model.predict(x_test), average="weighted")}')
        print(f'Mean length: {np.average(lengths)}')
        print(f'Accuracy: {1 - (times_wrong / len(x_test))}')
        print(f'Fidelity: {1 - (model_wrong / len(x_test))}')
        print(f'Time: {np.average(times)}')

        count += 1


def perform_tests_adult():
    c_file = ce.utils.download_data('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
    c_df = pd.read_csv(c_file, header=None, skipinitialspace=True)
    c_df = c_df.drop([2, 4], axis=1)

    # Give descriptive names to features
    c_features    = ['age', 'workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country']
    c_categorical = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country']
    c_df.columns  = c_features + ['class']
    c_contrasts = c_df['class'].unique()
    data, target = c_df.iloc[:,:-1], c_df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=SEED)

    random_forest = ensemble.RandomForestClassifier(random_state=SEED, n_estimators=500)
    logistic_regression = linear_model.LogisticRegression(random_state=SEED)
    SVM = svm.SVC(random_state=SEED, probability=True)
    nn = neural_network.MLPClassifier()

    models = [logistic_regression, nn]
    model_to_str = ['Random Forest', 'Logistic Regression', 'SVM', 'Neural Network']
    count = 0
    for model in models:
        model = Pipeline([('label_encoder', ce.CustomLabelEncoder(c_categorical).fit(data)),('classifier', model)])
        model.fit(x_train, y_train)
        dm = ce.domain_mappers.DomainMapperPandas(x_train, contrast_names=c_contrasts, categorical_features=[i for i in range(len(c_features)) if c_features[i] in c_categorical])
        exp = ce.ContrastiveExplanation(dm)
        times = np.array([])
        lengths = np.array([])
        times_wrong = 0
        model_wrong = 0
        test = np.array(np.array([39, 'Local-gov', '11th', 'Married-civ-spouse', 'Transport-moving', 'Husband', 'White', 'Male', 0, 0, 30, 'United-States'], dtype='<U32'), dtype='O').reshape(1,-1)
        for i in range(100):
            sample = x_test.iloc[i]
            model.predict_proba(test)
            label = y_test.iloc[i]
            start = timer()
            _, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            if len(counterfactuals) == 0:
                times_wrong += 1
            times = np.append(times, end - start)
            lengths = np.append(lengths, len(counterfactuals))
            if model[1].classes_[np.argmax(model.predict_proba(np.array(sample).reshape(1, -1)))] != label:
                model_wrong += 1
        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.f1_score(y_test, model.predict(x_test), average="weighted")}')
        print(f'Mean length: {np.average(lengths)}')
        print(f'Accuracy: {1 - (times_wrong / len(x_test))}')
        print(f'Fidelity: {1 - (model_wrong / len(x_test))}')
        print(f'Time: {np.average(times)}')

        count += 1

print('\nAdult')
perform_tests_adult()

print('\nIris')
perform_tests(iris)

# print('\nDiabetes')
# perform_tests_regression(diabetes)

print('\nHeart Disease')
perform_tests_pandas(heart)

print('\nAdult')
perform_tests_adult()
