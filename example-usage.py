import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import linear_model, naive_bayes, datasets, ensemble, svm, neural_network, metrics
import json
import contrastive_explanation as ce
from sklearn.pipeline import Pipeline

SEED = np.random.RandomState(2020)

iris = datasets.load_iris()
diabetes = datasets.load_diabetes()
heart = pd.read_csv('./heart.csv')

def roundt(val,digits):
   return round(val+10**(-len(str(val))-1), digits)

def adult_split(xs, ys, rule, fact, foil):
    # fact = rule[1]
    # foil = rule[2]
    # rule = rule[0]
    # index = rule[0][0]
    # value = rule[0][2]

    index = rule[0]
    value = rule[2]

    lefty = []
    righty = []

    for i in range(len(ys)):
        # if ys[i] == fact or ys[i] == foil:
            if isinstance(xs.iloc[i][index], str):
                if xs.iloc[i][index] == value:
                    # print('string', xs.iloc[i][index], value)
                    lefty.append(ys.iloc[i])
                else:
                    righty.append(ys.iloc[i])
            else:
                if xs.iloc[i][index] < value:
                    # print('int', xs.iloc[i][index], value)
                    lefty.append(ys.iloc[i])
                else:
                    righty.append(ys.iloc[i])
    return [lefty, righty]

def split(xs, ys, rule, fact, foil):
    # fact = rule[1]
    # foil = rule[2]
    #
    # rule = rule[0]
    # index = rule[0][0]
    # value = rule[0][2]

    index = rule[0]
    value = rule[2]

    lefty = []
    righty = []

    for i in range(len(ys)):
        # if ys[i] == fact or ys[i] == foil:
        #     print(index)
            if xs[i][index] < value:
                lefty.append(ys[i])
            else:
                righty.append(ys[i])
    return [lefty, righty]

#deprecated
def gini_index(yss, labels):
    # sum weighted Gini index for each group
    gini = 0.0
    for group in yss:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in labels:
            # print(group)
            # print(size)
            # print(group.count(class_val))
            p = group.count(class_val) / size
            score += p * p
            # print('score:', score)
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / len(yss[0] + yss[1]))
    return gini

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

def get_IG(ys, yss, foil):
    # ys = ys.tolist()
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
        #zeros for every rule
        lengths = [0, 0, 0, 0]
        times_wrongs = [0, 0, 0, 0]
        model_wrongs = [0, 0, 0, 0]
        total_scores = [[], [], [] ,[]]
        labels = dataset.target_names
        amount = len(x_test)
        for i in range(len(x_test)):
            sample = x_test[i]
            label = y_test[i]
            start = timer()
            _, rules, facts, fact, foil, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            fact = labels.tolist().index(fact)
            foil = labels.tolist().index(foil)
            #process per rule
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
                    full_rule = (rules[i], fact, foil)
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
            # print(i)
            # print(roundt(accuracy, 2), '& ', roundt(fidelity, 2), '& ', roundt(mean_length, 2), '& ', roundt(mean_score, 3), '& ', roundt(metric, 3))


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
        #zeros for every rule
        lengths = [0, 0, 0, 0]
        times_wrongs = [0, 0, 0, 0]
        model_wrong = 0
        total_scores = [0, 0, 0, 0]
        labels = dataset.target_names
        for i in range(len(x_test)):
            sample = x_test[i]
            label = y_test[i]
            start = timer()
            _, rules, fact, foil, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            fact = labels.tolist().index(fact)
            foil = labels.tolist().index(foil)
            #process per rule
            for i in range(len(rules)):
                full_rule = (rules[i], fact, foil)
                yss = split(x_test, y_test, full_rule)
                total_scores[i] += gini_index(yss, range(len(labels)))
                if len(counterfactuals[i]) == 0:
                    times_wrongs[i] += 1
                lengths1 = np.append(lengths1, len(rules[i]))
            times = np.append(times, end - start)
            if model.predict(sample.reshape(1, -1))[0] != label:
                model_wrong += 1
        print(f'\n{model_to_str[count]}')
        print(f'F1 score: {metrics.f1_score(y_test, model.predict(x_test), average="weighted")}')
        print(f'Fidelity: {1 - (model_wrong / len(x_test))}')
        print(f'Time: {np.average(times)}')
        for i in range(len(rules)):
            print(f'Mean length: {np.average(lengths[i])}')
            print(f'Accuracy: {1 - (times_wrongs[i] / len(x_test))}')
            print(f'Average score of explanations: {total_scores[i] / len(x_test)}')

        count += 1


def perform_tests_pandas(dataset):
    target = dataset['target']
    data = dataset.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=SEED)
    xs = x_test.to_numpy()
    ys = y_test.to_numpy()
    random_forest = ensemble.RandomForestClassifier(random_state=SEED, n_estimators=500)
    logistic_regression = linear_model.LogisticRegression(random_state=SEED)
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
        #zeros for every rule
        lengths = [0, 0, 0, 0]
        times_wrongs = [0, 0, 0, 0]
        model_wrongs = [0, 0, 0, 0]
        total_scores = [[], [], [], []]
        labels = y_test.unique()
        amount = len(x_test)
        for i in range(len(x_test)):
            sample = xs[i]
            label = ys[i]
            start = timer()
            _, rules, facts, fact, foil, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            fact = labels.tolist().index(fact)
            # print(fact)
            foil = labels.tolist().index(foil)
            # print(model.predict(sample.reshape(1, -1))[0])
            predict_index = model.predict(sample.reshape(1, -1))[0]
            #process per rule
            for i in range(len(rules)):
                if foil == predict_index:
                    if facts[i] != 1:
                        model_wrongs[i] += 1
                        # print('wrong')
                    else:
                        if facts[i] != 0:
                            model_wrongs[i] += 1
                            # print('wrong')
                if facts[i] == 1:
                    times_wrongs[i] += 1
                else:
                    full_rule = (rules[i], fact, foil)
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
            # print(i)
            # print(roundt(accuracy, 2), '& ', roundt(fidelity, 2), '& ', roundt(mean_length, 2), '& ', roundt(mean_score, 3), '& ', roundt(metric, 3))

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
    xs = x_test.reset_index(drop=True)
    ys = y_test.reset_index(drop=True)
    random_forest = ensemble.RandomForestClassifier(random_state=SEED, n_estimators=500)
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
        # zeros for every rule
        lengths = [0, 0, 0, 0]
        times_wrongs = [0, 0, 0, 0]
        model_wrongs = [0, 0, 0, 0]
        total_scores = [[], [], [], []]
        labels = y_test.unique()
        amount = 5
        print(len(x_test))
        for i in range(amount):
            print('processing', i, '/', amount, '...')
            sample = x_test.iloc[i]
            label = y_test.iloc[i]
            start = timer()
            _, rules, facts, fact, foil, counterfactuals = exp.explain_instance_domain(model.predict_proba, sample)
            end = timer()
            fact_index = 0
            if fact == '>50K':
                fact_index = 1
            foil_index = 0
            if foil == '>50K':
                foil_index = 1
            predict_index = 0
            if model[1].classes_[np.argmax(model.predict_proba(np.array(sample).reshape(1, -1)))] == '>50K':
                predict_index = 1
            # process per rule
            for i in range(len(rules)):
                print(rules[i])
                if foil_index == predict_index:
                    if facts[i] != 1:
                        model_wrongs[i] += 1
                    else:
                        if facts[i] != 0:
                            model_wrongs[i] += 1
                if facts[i] == 1:
                    times_wrongs[i] += 1
                else:
                    full_rule = (rules[i], fact, foil)
                    # print(full_rule)
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
            # print(i)
            # print(roundt(accuracy, 2), '& ', roundt(fidelity, 2), '& ', roundt(mean_length, 2), '& ', roundt(mean_score, 3), '& ', roundt(metric, 3))

        count += 1


print('\nIris')
perform_tests(iris)

# print('\nDiabetes')
# perform_tests_regression(diabetes)

print('\nHeart Disease')
# perform_tests_pandas(heart)

print('\nAdult')
# perform_tests_adult()