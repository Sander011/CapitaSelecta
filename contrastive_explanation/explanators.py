"""Uses generic tabular data to explain a single instance with
a contrastive/counterfactual explanation.

Attributes:
    DEBUG (bool): Debug mode enabled
"""

import numpy as np
import networkx as nx
import warnings

from sklearn import tree, ensemble, metrics
from sklearn.tree import _tree
from sklearn.utils import check_random_state
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from .rules import Operator, Literal
from .utils import cache, check_stringvar, check_relvar, print_binary_tree


DEBUG = False


class Explanator:
    """General class for Explanators (method to acquire explanation)."""

    def __init__(self,
                 verbose=False,
                 seed=1):
        """Init.

        Args:
            verbose (bool): Print intermediary steps of algorithm
            seed (int): Seed for random functions
        """
        self.verbose = verbose
        self.seed = check_random_state(seed)

    def get_rule(self,
                 fact_sample,
                 fact,
                 foil,
                 xs,
                 ys,
                 weights,
                 **kwargs):
        """Get rules for 'fact' and 'foil' using an explanator.

        Args:
            fact_sample: Sample x of fact
            fact: Outcome y = m(x) of fact
            foil: Outcome y for foil
            xs: Training data
            ys: Training data labels, has to contain
                observations with the foil
            weights: Weights of training data, based on
                distance to fact_sample
            foil_strategy: Strategy for finding the
                foil decision region ('closest', 'random')

        Returns:
            foil_path (descriptive_path for foil), confidence
        """
        raise NotImplementedError('Implemented in subclasses')

    def get_explanation(self, rules):
        """Get explanation given a set of rules."""
        raise NotImplementedError('Implemented in subclasses')


class RuleExplanator(Explanator):
    """General class for rule-based Explanators."""

    def get_explanation(self, rules, contrastive=True):
        """Get an explanation given a rule, of why the fact
        is outside of the foil decision boundary (contrastive) or
        why the fact is inside the fact decision boundary.
        """
        for feature, threshold, _, foil_greater, fact_greater in rules:
            if (contrastive and fact_greater and not foil_greater or
                    not contrastive and foil_greater):
                yield Literal(feature, Operator.GT, threshold)
            elif (contrastive and not fact_greater and foil_greater or
                    not contrastive and not foil_greater):
                yield Literal(feature, Operator.SEQ, threshold)
            else:
                yield None


class TreeExplanator(RuleExplanator):
    """Explain using a decision tree."""

    def __init__(self,
                 generalize=2,
                 verbose=False,
                 print_tree=False,
                 feature_names=[],
                 seed=1):
        """Init.

        Args:
            Generalize [0, 1]: Lower = overfit more, higher = generalize more
        """
        super().__init__(verbose=verbose, seed=seed)
        self.generalize = generalize
        self.tree = None
        self.graph = None
        self.print_tree = print_tree
        self.feature_names = feature_names

    @cache
    def _foil_tree(self, xs, ys, weights, seed, **dtargs):
        """Classifies foil-vs-rest using a DecisionTreeClassifier.

        Args:
            xs: Input data
            ys: Input labels (1 = foil, 0 = else)
            weights: Input sample weights
            **dtargs: Pass on additional arguments to
                    DecisionTreeClassifier

        Returns:
            Trained model on input data for binary
            classification (output vs rest)
        """
        model = tree.DecisionTreeClassifier(random_state=check_random_state(seed),
                                            class_weight='balanced',
                                            **dtargs)
        model.fit(xs, ys, sample_weight=weights)

        # If we only have a root node there is no explanation, so try acquiring
        # and explanation by training a forest of trees and picking the highest
        # performance estimator
        if model.tree_.max_depth < 2:
            seed_ = check_random_state(seed)
            forest = ensemble.RandomForestClassifier(random_state=seed_,
                                                     class_weight='balanced')
            forest.fit(xs, ys, sample_weight=weights)

            estimators = [(e.score(xs, ys), e) for e in forest.estimators_
                          if e.tree_.max_depth > 1]

            if estimators is not None and estimators:
                model = sorted(estimators, key=lambda x: x[0], reverse=True)[0][1]

        local_fidelity = metrics.accuracy_score(ys, model.predict(xs))

        if self.verbose:
            print('[E] Fidelity of tree on neighborhood data =', local_fidelity)

        if DEBUG:
            print_binary_tree(model, xs[0].reshape(1, -1), self.feature_names)

        return model, local_fidelity

    def descriptive_path(self, decision_path, sample, tree):
        """Create a descriptive path for a decision_path of node ids.

        Args:
            decision_path (list, np.array): Node ids to describe
            sample: Sample to describe
            tree: sklearn tree used to create decision_path

        Returns:
            Tuples (feature, threshold, sample value, greater,
                    decision_path > threshold,
                    sample value > threshold)
            for all node ids in the decision_path
        """
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        return [(feature[node],
                 threshold[node],
                 sample[feature[node]],
                 greater,
                 float(sample[feature[node]]) > threshold[node])
                for node, greater in decision_path]

    def decision_path(self, tree, sample):
        """Get a descriptive decision path of a sample.

        Args:
            tree: sklearn tree
            sample: Sample to decide decision path of

        Returns:
            Descriptive decision path for sample
        """
        dp = list(np.nonzero(tree.decision_path(sample.reshape(1, -1)))[1])
        if len(dp) == 0:
            return []
        turned_right = [dp[i] in tree.tree_.children_right
                        for i, node in enumerate(dp[:-1])] + [False]

        return self.descriptive_path(list(zip(dp, turned_right)), sample, tree)

    def __to_graph(self, t, node=0):
        """Recursively obtain graph of a sklearn tree.

        Args:
            t: sklearn tree.tree_
            node: Node ID

        Returns: Graph of tuples (parent_id, child_id, right_path_taken)
        """
        left = t.children_left[node]
        right = t.children_right[node]

        if left != _tree.TREE_LEAF:
            left_path = [(node, left, False)] + self.__to_graph(t, left)
            right_path = [(node, right, True)] + self.__to_graph(t, right)
            return left_path + right_path
        return []

    def __get_nodes(self, graph):
        nodes = []
        for g in graph:
            nodes.extend(g)
        return [n for n in list(set(nodes)) if n not in [True, False]]

    @cache
    def _fact_foil_graph(self, tree, start_node=0):
        """Convert a tree into a graph from the fact_leaf to
        all other leaves.

        Args:
            tree: sklearn tree.tree_
            start_node: Node ID to start constructing graph from

        Returns:
            Graph, list of foil nodes
        """
        # Convert tree to graph
        graph = self.__to_graph(tree, node=start_node)

        # Acquire the foil leafs
        foil_nodes = [node for node in self.__get_nodes(graph)
                      if (tree.feature[node] == _tree.TREE_UNDEFINED and
                          np.argmax(tree.value[node]) == 1)]

        return graph, foil_nodes

    def __construct_tuples(self, graph, tree_data, strategy='informativeness'):
        for v1, v2, greater in graph:
            if strategy == 'closest':
                yield v1, v2, greater, 1.0
            elif strategy == 'size':
                yield v1, v2, greater, 1 - (tree_data.n_node_samples[v2] /
                                            sum(tree_data.n_node_samples))
            elif strategy == 'impurity':
                yield v1, v2, greater, 1 - abs(tree_data.impurity[v1] -
                                               tree_data.impurity[v2])
            elif strategy == 'informativeness':
                yield v1, v2, greater, (1 / abs(tree_data.impurity[v1] -
                                                tree_data.impurity[v2]) +
                                        1 / tree_data.n_node_samples[v2])
            elif strategy == 'random':
                yield v1, v2, greater, np.random.random_sample()
            else:
                yield v1, v2, greater, 0.0

    def __shortest_path(self, g, start, end):
        """Determine shortest path from 'start' to
        'end' in undirected graph 'g'.

        Args:
            g: Graph represented using list of tuples
                (vertex1, vertex2, _, vertex_weight)
            start: Start vertex
            end: End vertex

        Returns:
            Shortest path (list of vertices)
        """
        G = nx.Graph()
        for v1, v2, _, w in g:
            G.add_edge(v1, v2, weight=w)
        return nx.shortest_path(G, start, end, weight='weight')

    @check_stringvar(('strategy', ['closest', 'size', 'impurity',
                                   'informativeness', 'random']))
    def _get_path(self,
                  graph,
                  fact_node,
                  foil_nodes,
                  tree_data,
                  strategy='informativeness'):
        """Get shortest path in graph based on strategy.

        Args:
            graph: Unweighted graph with tuples (v1, v2, _)
                reconstructed from decision tree.
            fact_node: Leaf node 'fact_sample' ended up in
            foil_nodes: List of nodes with decision foil
            tree_data: sklearn.tree.tree_
            strategy: Weight strategy (see 'get_rules()')

        Returns:
            List of foil decisions, represented as descriptive_path
        """
        # Add weights to vertices
        weighted_graph = list(self.__construct_tuples(graph, tree_data,
                                                      strategy))

        # Add final point '-1' to find shortest path to, add 0 weight edge
        foil_sink = -1
        final_graph = np.array(weighted_graph + [(f, foil_sink, False, 0.0)
                                                 for f in foil_nodes],
                               dtype=np.dtype([('v1', 'int'),
                                               ('v2', 'int'),
                                               ('greater', 'bool'),
                                               ('w', 'float')]))

        # Get shortest path
        shortest_path = self.__shortest_path(final_graph,
                                             fact_node,
                                             foil_sink)[:-1]

        # Get confidence (accuracy of foil leaf)
        foil_leaf_classes = tree_data.value[shortest_path[-1]]
        confidence = foil_leaf_classes[0, 1] / np.sum(foil_leaf_classes)

        if self.verbose:
            print(f'[E] Found shortest path {shortest_path} using '
                  f'strategy "{strategy}"')

        # Decisions taken for path
        foil_decisions = []
        for a, b in zip(shortest_path[:-1], shortest_path[1:]):
            for edge in final_graph:
                if a == edge[0] and b == edge[1]:
                    foil_decisions.append((edge[0], edge[2]))

        return foil_decisions, confidence

    @check_relvar(('beta', '>= 1'))
    def closest_decision(self, tree, sample,
                         strategy='informativeness',
                         beta=5):
        """Find the closest decision that is of a class other than the
        target class.

        Args:
            tree: sklearn tree
            sample: Entry to explain
            beta: Hyperparameter >= 1 to determine when to only
                search part of tree (higher = search smaller area)

        Returns:
            Ordered descriptive decision path difference,
            confidence of leaf decision
        """
        # Only search part of tree depending on tree size
        decision_path = tree.decision_path(sample.reshape(1, -1)).indices
        if len(decision_path) < 2:
            warnings.warn('Stub tree')
            return None, 0.0
        start_depth = int(round(len(decision_path) / beta))
        start_node = decision_path[start_depth]

        # Get decision for sample
        fact_leaf = tree.apply(sample.reshape(1, -1)).item(0)

        # TODO: Retrain tree if wrong prediction
        # if np.argmax(tree.tree_.value[fact_leaf]) != 0:
        #     warnings.warn('Tree did not predict as fact')

        # Find closest leaf that does not predict output x, based on a strategy
        graph, foil_nodes = self._fact_foil_graph(tree.tree_,
                                                  start_node=start_node)

        if self.verbose:
            print(f'[E] Found {len(foil_nodes)} contrastive decision regions, '
                  f'starting from node {start_node}')

        if len(foil_nodes) == 0:
            return None, 0

        # Contrastive decision region
        foil_path, confidence = self._get_path(graph,
                                               fact_leaf,
                                               foil_nodes,
                                               tree.tree_,
                                               strategy)

        return self.descriptive_path(foil_path, sample, tree), confidence

    """ New code starts here """
    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Determine whether all labels in dataset have same label
    def pure_node(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        return class_values[1:] == class_values[:-1]

    # Select the best split point for a dataset
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        # check for a no split
        left, right = b_groups
        if not left or not right:
            return self.to_terminal(left + right)
        return {'index': b_index, 'value': b_value, 'groups': b_groups, 'score': b_score}

    # create split point for a dataset based on predefined split
    def get_path_split(self, dataset, path):
        if len(path) == 0:
            return self.to_terminal(dataset), []
        while len(path) > 0:
            node = path[0]
            b_index = node['index']
            b_value = node['value']
            b_groups = self.test_split(b_index, b_value, dataset)
            class_values = list(set(row[-1] for row in dataset))
            b_score = self.gini_index(b_groups, class_values)
            left, right = b_groups
            if not left or not right:
                if len(path) > 0:
                    # print('skipping obsolete decision')
                    path = path[1:]
                else:
                    return self.to_terminal(left + right)
            else:
                return {'index': b_index, 'value': b_value, 'groups': b_groups, 'score': b_score}, path
        return {'index': b_index, 'value': b_value, 'groups': b_groups, 'score': b_score}, path

    # Create child splits for a node or make terminal
    def path_split(self, node, max_depth, min_size, depth, path):
        if isinstance(node, dict):
            left, right = node['groups']
            del (node['groups'])
            # check for a no split
            # check for max depth
            if depth >= max_depth:
                node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
                return
            # check path
            if len(path) > 0:
                if path[0]['side'] == 'left':
                    path = path[1:]
                    # pathsplit left
                    # print('going left')
                    if self.pure_node(left):
                        # print('pure node found')
                        node['left'] = self.to_terminal(left)
                    else:
                        node['left'], path = self.get_path_split(left, path)
                        if isinstance(node['left'], dict):
                            # print('node is dict')
                            self.path_split(node['left'], max_depth, min_size, depth + 1, path)
                    # process right child
                    if len(right) <= min_size or self.pure_node(right):
                        node['right'] = self.to_terminal(right)
                    else:
                        node['right'] = self.get_split(right)
                        if isinstance(node['right'], dict):
                            # print('node is dict')
                            self.split(node['right'], max_depth, min_size, depth + 1)
                elif path[0]['side'] == 'right':
                    path = path[1:]
                    # pathsplit right
                    # print('going right')
                    if self.pure_node(right):
                        # print('pure node found')
                        node['right'] = self.to_terminal(right)
                    else:
                        node['right'], path = self.get_path_split(right, path)
                        if isinstance(node['right'], dict):
                            self.path_split(node['right'], max_depth, min_size, depth + 1, path)
                    # process left child
                    if len(left) <= min_size or self.pure_node(left):
                        node['left'] = self.to_terminal(left)
                    else:
                        node['left'] = self.get_split(left)
                        if isinstance(node['left'], dict):
                            self.split(node['left'], max_depth, min_size, depth + 1)
                else:
                    print('unknown side in path')


    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size or self.pure_node(left):
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            if isinstance(node['left'], dict):
                self.split(node['left'], max_depth, min_size, depth + 1)
        # process right child
        if len(right) <= min_size or self.pure_node(right):
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            if isinstance(node['right'], dict):
                self.split(node['right'], max_depth, min_size, depth + 1)

    # Build a decision tree
    def build_tree(self, train, max_depth, min_size):
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

    # Build a decision tree with a specified path
    def build_path_tree(self, train, max_depth, min_size, path):
        root, path = self.get_path_split(train, path)
        self.path_split(root, max_depth, min_size, 1, path)
        return root

    # Print a decision tree
    def print_tree_func(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * '   ', (node['index']), node['value'])))
            print(depth * '   ', 'gini index:', node['score'])
            self.print_tree_func(node['left'], depth + 1)
            self.print_tree_func(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * '   ', node)))

    def predict(self, sample, node):
        if isinstance(node, dict):
            if sample[node['index']] < node['value']:
                return self.predict(sample, node['left'])
            else:
                return self.predict(sample, node['right'])
        else:
            return node

    def get_path(self, sample, node):
        return self.predict_path(sample, node, [])

    def predict_path(self, sample, node, path):
        if isinstance(node, dict):
            if sample[node['index']] < node['value']:
                return self.predict_path(sample, node['left'], path + [
                    {'index': node['index'], 'value': node['value'], 'score': node['score'], 'side': 'left'}])
            else:
                return self.predict_path(sample, node['right'], path + [
                    {'index': node['index'], 'value': node['value'], 'score': node['score'], 'side': 'right'}])
        else:
            return path

    def get_train(self, xs, ys):
        train = []
        xslist = xs.tolist()
        yslist = ys.tolist()
        for i in range(len(xslist)):
            train.append(xslist[i] + [yslist[i]])
        return train

    def get_tree(self, xs, ys):
        train = self.get_train(xs, ys)
        return self.build_tree(train, 100, 1)

    def add_samples_to_tree(self, samples, node):
        for sample in samples:
            self.add_to_tree(sample, node)
        return node

    def add_size(self, node):
        node['amount'] = 0
        node['samples'] = []
        if isinstance(node['left'], dict):
            self.add_size(node['left'])
        else:
            node['left'] = [node['left'], 0, [], 0]
        if isinstance(node['right'], dict):
            self.add_size(node['right'])
        else:
            node['right'] = [node['right'], 0, [], 0]

    def add_to_tree(self, sample, node):
        if isinstance(node, dict):
            node['amount'] += 1
            node['samples'].append(sample)
            if sample[node['index']] < node['value']:
                self.add_to_tree(sample, node['left'])
            else:
                self.add_to_tree(sample, node['right'])
        else:
            node[1] += 1
            node[2].append(sample)

    # Print a decision tree with samples
    def print_sample_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * '   ', (node['index']), node['value'])))
            print(depth * '   ', 'gini index:', node['score'])
            print(depth * '   ', 'amount:', node['amount'])
            self.print_sample_tree(node['left'], depth + 1)
            self.print_sample_tree(node['right'], depth + 1)
        else:
            print(depth * '   ', 'label:', node[0], 'amount:', node[1])

    def get_entropy(self, ys):
        ratio = sum(ys) / len(ys)
        if ratio == 1 or ratio == 0:
            return 0
        else:
            return -(ratio * np.log2(ratio) + (1 - ratio) * np.log2(1 - ratio))

    def get_IG(self, index, value, dataset):
        ys = self.get_ys(dataset)
        entr = self.get_entropy(ys)
        left, right = self.test_split(index, value, dataset)
        if not left or not right:
            return 0
        left_ratio = len(left) / len(ys)
        right_ratio = 1 - left_ratio
        IG = entr - (left_ratio * self.get_entropy(self.get_ys(left)) + right_ratio * self.get_entropy(
            self.get_ys(right)))
        return IG

    # gets labels if placed at end of samples
    def get_ys(self, dataset):
        y = len(dataset[0]) - 1
        return [x[y] for x in dataset]

    # if all weights are 1, the shortest path is the closest foil leaf method
    def add_weight_1(self, node, total):
        if isinstance(node, dict):
            weight = 1
            node['weight'] = weight
            self.add_weight_1(node['left'], total)
            self.add_weight_1(node['right'], total)

    # adds the weight to a given node's edges
    def add_weight(self, node, total):
        if isinstance(node, dict):
            IG = self.get_IG(node['index'], node['value'], node['samples'])
            node['IG'] = IG
            if IG == 0:
                weight = np.inf
            else:
                weight = 1 / ((node['amount'] / total) * node['IG'])
            node['weight'] = weight
            self.add_weight(node['left'], total)
            self.add_weight(node['right'], total)

    # Print a decision tree with samples
    def print_weight_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * '   ', (node['index']), node['value'])))
            print(depth * '   ', 'gini index:', node['score'])
            print(depth * '   ', 'amount:', node['amount'])
            print(depth * '   ', 'weight:', node['weight'])
            self.print_weight_tree(node['left'], depth + 1)
            self.print_weight_tree(node['right'], depth + 1)
        else:
            print(depth * '   ', 'label:', node[0], 'amount:', node[1])

    # wrapper function for part of the result path to the fact leaf
    def fact_path(self, sample, node):
        return self.weighted_path(sample, node, [])

    # get path to fact with weighted edges
    def weighted_path(self, sample, node, path):
        if isinstance(node, dict):
            if sample[node['index']] < node['value']:
                return self.weighted_path(sample, node['left'], path + [
                    {'index': node['index'], 'value': node['value'], 'score': node['score'],
                     'weight': node['weight'],
                     'side': 'left'}])
            else:
                return self.weighted_path(sample, node['right'], path + [
                    {'index': node['index'], 'value': node['value'], 'score': node['score'],
                     'weight': node['weight'],
                     'side': 'right'}])
        else:
            return path

    # get total weight to a sample
    def get_path_weight(self, sample, node):
        total = 0
        if isinstance(node, dict):
            if sample[node['index']] < node['value']:
                total += self.get_path_weight(sample, node['left'])
            else:
                total += self.get_path_weight(sample, node['right'])
            total += node['weight']
        return total

    # get part of the result path leading to the foil leaf
    def foil_path(self, node, label):
        if isinstance(node, dict):
            left_weight, left_path = self.foil_path(node['left'], label)
            right_weight, right_path = self.foil_path(node['right'], label)
            if left_weight < right_weight:
                return node['weight'] + left_weight, [
                    {'index': node['index'], 'value': node['value'], 'score': node['score'],
                     'weight': node['weight'],
                     'side': 'left'}] + left_path
            else:
                return node['weight'] + right_weight, [
                    {'index': node['index'], 'value': node['value'], 'score': node['score'],
                     'weight': node['weight'],
                     'side': 'right'}] + right_path
        elif node[0] == label:
            return np.inf, []
        else:
            return 0, []

    #  combine fact and foil path to calculate the shortest path
    def get_full_path(self, sample, node, label):
        if sample[node['index']] < node['value']:
            fact_weight = self.get_path_weight(sample, node)
            fact_path = self.fact_path(sample, node)[::-1]
            foil_weight, foil_path = self.foil_path(node['right'], label)
            current_foil_weight = node['weight']
            current_foil_path = [
                {'index': node['index'], 'value': node['value'], 'score': node['score'], 'weight': node['weight'],
                 'side': 'right'}]
        else:
            fact_weight = self.get_path_weight(sample, node)
            fact_path = self.fact_path(sample, node)[::-1]
            foil_weight, foil_path = self.foil_path(node['left'], label)
            current_foil_weight = node['weight']
            current_foil_path = [
                {'index': node['index'], 'value': node['value'], 'score': node['score'], 'weight': node['weight'],
                 'side': 'left'}]
        total_weight = fact_weight + current_foil_weight + foil_weight
        total_path = fact_path + current_foil_path + foil_path
        return total_weight, total_path

    # return resulting foil path and the total weight of the foil path
    def get_foil_path(self, sample, node, label):
        if isinstance(node, dict):
            if sample[node['index']] < node['value']:
                weight, path = self.get_foil_path(sample, node['left'], label)
            else:
                weight, path = self.get_foil_path(sample, node['right'], label)
            current_weight, current_path = self.get_full_path(sample, node, label)
        else:
            return np.inf, []
        if weight < current_weight:
            return weight, path
        else:
            return current_weight, current_path

    def get_rule(self,
                 fact_sample,
                 fact,
                 foil,
                 xs,
                 ys,
                 weights,
                 foil_strategy='informativeness'):
        """Get rules for 'fact' and 'foil' using a
        decision tree explanator. For arguments see
        Explanator.get_rule().
        """

        normal_tree = self.get_tree(xs, ys)
        train = self.get_train(xs, ys)
        p = self.get_path(fact_sample, normal_tree)
        p = p[::-1]
        fact_normal = self.predict(fact_sample, normal_tree)
        reverse_tree = self.build_path_tree(train, 100, 1, p)
        normal_weighted_path = self.get_rule_path(train, normal_tree, fact_sample, len(ys))[1]
        normal_weighted_path = self.format_rule_path(normal_weighted_path, fact_sample)
        normal_shortest_path = self.get_rule_path_1(train, normal_tree, fact_sample, len(ys))[1]
        normal_shortest_path = self.format_rule_path(normal_shortest_path, fact_sample)
        fact_reverse = self.predict(fact_sample, reverse_tree)
        reverse_weighted_path = self.get_rule_path(train, reverse_tree, fact_sample, len(ys))[1]
        reverse_weighted_path = self.format_rule_path(reverse_weighted_path, fact_sample)
        reverse_shortest_path = self.get_rule_path_1(train, reverse_tree, fact_sample, len(ys))[1]
        reverse_shortest_path = self.format_rule_path(reverse_shortest_path, fact_sample)
        fidelity = self.get_fidelity(reverse_tree, train, normal_tree)

        r = [normal_shortest_path, normal_weighted_path, reverse_shortest_path, reverse_weighted_path]
        facts = [fact_normal, fact_normal, fact_reverse, fact_reverse]
        return r, facts, 1, fidelity

    def get_fidelity(self, tree1, xs, tree2):
        res = 0
        for x in xs:
            if self.predict(x, tree1)[0] == self.predict(x, tree2)[0]:
                res += 1
        if res == 0:
            return 0
        else:
            return res/len(xs)

    # format rule path for the rest of the code
    def format_rule_path(self, path, sample):

        return[(node['index'],
                node['value'],
                0,
                node['side'] == 'right',
                sample[node['index']] > node['value'])
               for node in path]

    # wrapper function for putting weights into a tree and getting the foil path
    def get_rule_path(self, dataset, tree, fact_sample, total_samples):
        self.add_size(tree)
        self.add_samples_to_tree(dataset, tree)
        tree['weight'] = 0
        self.add_weight(tree, total_samples)
        fact_label = self.predict(fact_sample, tree)
        path = self.get_foil_path(fact_sample, tree, fact_label)
        return path

    # wrapper function for putting weights into a tree and getting the foil path with closest leaf selection method
    def get_rule_path_1(self, dataset, tree, fact_sample, total_samples):
        self.add_size(tree)
        self.add_samples_to_tree(dataset, tree)
        tree['weight'] = 1
        self.add_weight_1(tree, total_samples)
        fact_label = self.predict(fact_sample, tree)
        path = self.get_foil_path(fact_sample, tree, fact_label)
        return path

class PointExplanator(Explanator):
    """Explain by selecting and comparing to a prototype point."""

    @check_stringvar(('strategy', ['closest', 'medoid', 'random']))
    def contrastive_prototype(self,
                              xs,
                              ys,
                              weights,
                              strategy='closest'):
        """Get a contrastive sample based on strategy."""
        # Get foil xs
        ys_slice = [idx for idx, y in enumerate(ys) if y == 1]
        xs_foil = xs[ys_slice]

        if xs_foil is None:
            return None

        if strategy == 'closest':
            return xs_foil[np.argmax(weights[1:]) + 1]
        elif strategy == 'medoid':
            print(xs_foil)
            return xs_foil[0][0]
        elif strategy == 'random':
            return xs_foil[np.random.randint(xs_foil.shape[0], size=1), :][0]

    def path_difference(self,
                        fact_sample,
                        foil_sample,
                        normalize=False):
        """Calculate difference between two equal length samples.

        Args:
            fact_sample: Sample for fact
            foil_sample: Sample for foil
            normalize (bool): TODO

        Returns:
            Difference between fact_sample and foil_sample ordered
            by magnitude of difference
        """
        if len(fact_sample) != len(foil_sample):
            raise Exception('Number of features of fact sample and '
                            'prototype point should be equal')

        difference = fact_sample - foil_sample
        difference_path = [(i, abs(d), fact_sample[i], d < 0)
                           for i, d in enumerate(difference)]

        # Sort by magnitude of difference
        return sorted(difference_path, key=lambda d: d[1], reverse=True)

    def get_rule(self,
                 fact_sample,
                 fact,
                 foil,
                 xs,
                 ys,
                 weights,
                 foil_strategy='closest',
                 **kwargs):
        """Get rules for 'fact' and 'foil' using a
        point explanator. For arguments see Explanator.get_rule().
        """
        if self.verbose:
            print("[E] Explaining with a prototype point...")

        # Acquire prototype for foil
        foil_sample = self.contrastive_prototype(xs, ys, weights,
                                                 strategy=foil_strategy)
        if foil_sample is None:
            return None, 0

        if self.verbose:
            print(f'[E] Found prototype point {foil_sample} using '
                  f'strategy "{foil_strategy}"')

        # Explain difference as path
        return self.path_difference(fact_sample, foil_sample), 0, 0

    def get_explanation(self, rules, contrastive=True):
        """Get an explanation given a rule, of why the fact
        is not a foil (contrastive) or why it is a fact.
        """
        for feature, difference, _, is_negative in rules:
            if (contrastive and is_negative or
                    not contrastive and not is_negative):
                yield Literal(feature, Operator.MINUSEQ, difference)
            else:
                yield Literal(feature, Operator.PLUSEQ, difference)
