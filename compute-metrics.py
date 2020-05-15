import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets, ensemble, svm, neural_network, metrics
import contrastive_explanation as ce
from sklearn.pipeline import Pipeline

SEED = np.random.RandomState(2020)

iris = datasets.load_iris()
diabetes = datasets.load_diabetes()
heart = pd.read_csv('./heart.csv')

# Round of to two decimals
def roundt(val,digits):
   return round(val+10**(-len(str(val))-1), digits)

# split adult set in two sets based on rule
def adult_split(xs, ys, rule, fact, foil):
    index = rule[0]
    value = rule[2]

    lefty = []
    righty = []

    for i in range(len(ys)):
            if isinstance(xs.iloc[i][index], str):
                if xs.iloc[i][index] == value:
                    lefty.append(ys.iloc[i])
                else:
                    righty.append(ys.iloc[i])
            else:
                if xs.iloc[i][index] < value:
                    lefty.append(ys.iloc[i])
                else:
                    righty.append(ys.iloc[i])
    return [lefty, righty]

# split set into two sets based on rule
def split(xs, ys, rule, fact, foil):

    index = rule[0]
    value = rule[2]

    lefty = []
    righty = []

    for i in range(len(ys)):
            if xs[i][index] < value:
                lefty.append(ys[i])
            else:
                righty.append(ys[i])
    return [lefty, righty]

# get entropy of set
def get_entropy(ys, foil):
    foils = 0
    for y in ys:
        if y == foil:
            foils += 1
        ratio = foils / len(ys)
    if ratio == 1 or ratio == 0:
        return 0
    else:
        return -(ratio * np.log2(ratio) + (1 - ratio) * np.log2(1 - ratio))

# get Information Gain of set ys split into sets yss
def get_IG(ys, yss, foil):
    entr = get_entropy(ys, foil)
    left = yss[0]
    right = yss[1]
    if not left or not right:
        return 0
    left_ratio = len(left) / len(ys)
    right_ratio = 1 - left_ratio
    IG = entr - (left_ratio * get_entropy(left, foil) + right_ratio * get_entropy(
        right, foil))
    return IG

# used for Iris set
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
        lengths = [0, 0, 0, 0]
        times_wrongs = [0, 0, 0, 0]
        model_wrongs = [0, 0, 0, 0]
        total_scores = [[], [], [] ,[]]
        labels = dataset.target_names
        amount = len(x_test)
        for i in range(len(x_test)):
            sample = x_test[i]
            start = timer()
            _, rules, facts, fact, foil, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            fact = labels.tolist().index(fact)
            foil = labels.tolist().index(foil)
            predict_index = model.predict(sample.reshape(1, -1))[0]
            for i in range(len(rules)):
                if foil == predict_index:
                    if facts[i] != 1:
                        model_wrongs[i] += 1
                    else:
                        if facts[i] != 0:
                            model_wrongs[i] += 1
                if facts[i] == 1:
                    times_wrongs[i] += 1
                else:
                    for decision in rules[i]:
                        yss = split(x_test, y_test, decision, fact, foil)
                        total_scores[i].append(get_IG(y_test, yss, foil))
                    lengths[i] += len(rules[i])

            times = np.append(times, end - start)

        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.f1_score(y_test, model.predict(x_test), average="weighted")}')
        print(f'Time: {np.average(times)}')
        for i in range(len(rules)):
            mean_length = lengths[i] / (amount - times_wrongs[i])
            mean_score = sum(total_scores[i]) / len(total_scores[i])
            accuracy = 1 - (times_wrongs[i] / amount)
            fidelity = 1 - (model_wrongs[i] / amount)
            metric = mean_score/mean_length
            print(f'Mean length: {mean_length}')
            print(f'Accuracy: {accuracy}')
            print(f'Average score of explanations: {mean_score}')
            print(f'Fidelity: {fidelity}')
            print(f'score divided by length: {metric}')

        count += 1

#  used for Heart Disease set
def perform_tests_pandas(dataset):
    target = dataset['target']
    data = dataset.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=SEED)
    xs = x_test.to_numpy()
    ys = y_test.to_numpy()
    SVM = svm.SVC(random_state=SEED, probability=True)
    nn = neural_network.MLPClassifier()

    models = [SVM, nn]
    model_to_str = ['SVM', 'Neural Network']
    count = 0
    for model in models:
        model.fit(x_train, y_train)
        dm = ce.domain_mappers.DomainMapperPandas(x_train)
        exp = ce.ContrastiveExplanation(dm)
        times = np.array([])
        lengths = [0, 0, 0, 0]
        times_wrongs = [0, 0, 0, 0]
        model_wrongs = [0, 0, 0, 0]
        total_scores = [[], [], [], []]
        labels = y_test.unique()
        amount = len(x_test)
        for i in range(len(x_test)):
            sample = xs[i]
            start = timer()
            _, rules, facts, fact, foil, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            fact = labels.tolist().index(fact)
            foil = labels.tolist().index(foil)
            predict_index = model.predict(sample.reshape(1, -1))[0]
            for i in range(len(rules)):
                if foil == predict_index:
                    if facts[i] != 1:
                        model_wrongs[i] += 1
                    else:
                        if facts[i] != 0:
                            model_wrongs[i] += 1
                if facts[i] == 1:
                    times_wrongs[i] += 1
                else:
                    for decision in rules[i]:
                        yss = split(xs, ys, decision, fact, foil)
                        total_scores[i].append(get_IG(y_test, yss, foil))
                    lengths[i] += len(rules[i])
            times = np.append(times, end - start)

        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.f1_score(y_test, model.predict(x_test), average="weighted")}')
        print(f'Time: {np.average(times)}')
        for i in range(len(rules)):
            mean_length = lengths[i] / (amount - times_wrongs[i])
            mean_score = sum(total_scores[i]) / len(total_scores[i])
            accuracy = 1 - (times_wrongs[i] / amount)
            fidelity = 1 - (model_wrongs[i] / amount)
            metric = mean_score/mean_length
            print(f'Mean length: {mean_length}')
            print(f'Accuracy: {accuracy}')
            print(f'Average score of explanations: {mean_score}')
            print(f'Fidelity: {fidelity}')
            print(f'score divided by length: {metric}')

        count += 1

# Used for adult set
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
    SVM = svm.SVC(random_state=SEED, probability=True)

    models = [SVM]
    model_to_str = ['SVM']
    count = 0
    for model in models:
        model = Pipeline([('label_encoder', ce.CustomLabelEncoder(c_categorical).fit(data)), ('classifier', model)])
        model.fit(x_train, y_train)
        dm = ce.domain_mappers.DomainMapperPandas(x_train, contrast_names=c_contrasts, categorical_features=[i for i in range(len(c_features)) if c_features[i] in c_categorical])
        exp = ce.ContrastiveExplanation(dm)
        times = np.array([])
        lengths = [0, 0, 0, 0]
        times_wrongs = [0, 0, 0, 0]
        model_wrongs = [0, 0, 0, 0]
        total_scores = [[], [], [], []]
        amount = 5
        print(len(x_test))
        for i in range(amount):
            print('processing', i, '/', amount, '...')
            sample = x_test.iloc[i]
            start = timer()
            _, rules, facts, fact, foil, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            foil_index = 0
            if foil == '>50K':
                foil_index = 1
            predict_index = 0
            if model[1].classes_[np.argmax(model.predict_proba(np.array(sample).reshape(1, -1)))] == '>50K':
                predict_index = 1
            for i in range(len(rules)):
                if foil_index == predict_index:
                    if facts[i] != 1:
                        model_wrongs[i] += 1
                    else:
                        if facts[i] != 0:
                            model_wrongs[i] += 1
                if facts[i] == 1:
                    times_wrongs[i] += 1
                else:
                    for decision in rules[i]:
                        yss = adult_split(x_test, y_test, decision, fact, foil)
                        total_scores[i].append(get_IG(y_test, yss, foil))
                    lengths[i] += len(rules[i])
            times = np.append(times, end - start)


        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.f1_score(y_test, model.predict(x_test), average="weighted")}')
        print(f'Time: {np.average(times)}')
        for i in range(len(rules)):
            mean_length = lengths[i] / (amount - times_wrongs[i])
            mean_score = sum(total_scores[i]) / len(total_scores[i])
            accuracy = 1 - (times_wrongs[i] / amount)
            fidelity = 1 - (model_wrongs[i] / amount)
            metric = mean_score/mean_length
            print(f'Mean length: {mean_length}')
            print(f'Accuracy: {accuracy}')
            print(f'Average score of explanations: {mean_score}')
            print(f'Fidelity: {fidelity}')
            print(f'score divided by length: {metric}')

        count += 1


print('\nIris')
perform_tests(iris)

print('\nHeart Disease')
perform_tests_pandas(heart)

print('\nAdult')
perform_tests_adult()