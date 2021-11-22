# *******

import numpy as np
from utils.Loss import Entropy, Gini, MeanSquareError

class exact_split_search():
    """
    class spliiter candidate proposal
    Args:
        :X:
        :y:
    return:
        iterable Object
    """


    def __init__(self, X,y):
        self.X = X
        self.y = y
        self.m= X.shape[1]
        self.n=X.shape[0]
    def __iter__(self):
        self.feature_idx = 0
        self.threshold_idx = -1
        return self
    def __next__(self):
        self.X_candidate = np.unique(self.X[:,self.feature_idx])
        self.X_candidate_n = len(self.X_candidate)
        self.threshold_idx += 1
        if self.threshold_idx == self.X_candidate_n:
            self.threshold_idx =0
            self.feature_idx += 1
        if self.feature_idx == self.m:
            raise StopIteration
        return self.feature_idx , self.X_candidate[self.threshold_idx]

class quantile_split_search():
    """
    class spliiter candidate proposal
    Args:
        :X:
        :y:
        :n_proposal: number of proposal for each step and each feature
        :random_state:
    return:
        iterable Object
    """
    def __init__(self, X,y, n_proposal=20, random_state = None):

        self.X = X
        self.y = y
        self.random_state = random_state
        self.n_proposal = n_proposal
        self.m= X.shape[1]
        self.n=X.shape[0]
        np.random.seed(random_state)
    def __iter__(self):
        self.feature_idx = 0
        self.threshold_idx = -1
        return self
    def __next__(self):
        self.X_candidate = np.linspace(self.X[:,self.feature_idx].min(),
                                      self.X[:,self.feature_idx].max(),self.n_proposal)
        np.random.shuffle(self.X_candidate)
        self.X_candidate_n = len(self.X_candidate)
        self.threshold_idx += 1
        if self.threshold_idx == self.X_candidate_n:
            self.threshold_idx =0
            self.feature_idx += 1
        if self.feature_idx == self.m:
            raise StopIteration
        return self.feature_idx , self.X_candidate[self.threshold_idx]


class TreeNode():
    """
    class Tree node
    Args:
        :feature_idx: spliter index
        :threshold: spliter threshold
        :depth: depth of the node in tree
        :is_leaf: bool, if is a leaf node

    method:
        activate:
            activate the node
            args:
                :X:
                :y:

        predict:
            predict an instance
            args:
                :x:
    """
    def __init__(self, feature_idx=None, threshold=None, depth=None,is_leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.depth = depth
        self.left = None
        self.right = None

    def activate(self,X,y):

        if self.is_leaf:
            self.values, self.counts = np.unique(y,return_counts=True)
            return (None, None, None, None)

        X_left = X[X[:,self.feature_idx]< self.threshold]
        y_left = y[X[:,self.feature_idx]< self.threshold]
        X_right = X[X[:,self.feature_idx] >= self.threshold]
        y_right = y[X[:,self.feature_idx] >= self.threshold]

        return X_left, y_left, X_right, y_right

    def predict(self,x):
        if self.is_leaf:
            return self.values[self.counts.argmax()]
        else:
            if x[self.feature_idx] <= self.threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)


class DecisionTree():
    """
    class Decision Tree

    Args:


    """
    def __init__(self, criterion='entropy', splitter= 'exact', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    def _calculate_loss_decrease(self, y, y_left, y_right):
        if self.criterion == 'entropy':
            loss = Entropy()
        if self.criterion == 'gini':
            loss = Gini()
        gain = loss.loss(y)\
                -loss.loss(y_left)*len(y_left)/len(y)\
                -loss.loss(y_right)*len(y_right)/len(y)

        return gain

    def _best_splitter(self,X,y):
        score = 0
        splitter = dict(feature_idx=None,
                        threshold=None)
        if self.splitter == 'exact':
            proposal = exact_split_search(X,y)
        elif self.splitter == 'quantile':
            proposal =  quantile_split_search(X,y,random_state=self.random_state)

        for feature_idx, threshold in proposal:

            score_new = self._calculate_loss_decrease(y,
                                                 y[X[:,feature_idx]< threshold],
                                                 y[X[:,feature_idx]>= threshold])
            if score < score_new:
                score = score_new
                splitter = dict(feature_idx=feature_idx,
                                threshold=threshold)

        return splitter, score

    def _tree_grow(self,node,X,y):
        """
        Recursion Function for tree growing
        _tree_grow(parent)
        |-- _tree_grow(left)
        |-- _tree_grow(right)
        Args:
            :node: Object RegressionNode
            :X:
            :y:
        """

        X_left, y_left, X_right, y_right =  node.activate(X,y)

        if not node.is_leaf:
            if node.depth == self.max_depth:
                is_leaf = True
            else:
                is_leaf = False

            left_spliter, score_left = self._best_splitter(X_left,y_left)
            is_leaf_left = is_leaf or len(np.unique(y_left)) == 1 or score_left ==0

            node.left = TreeNode(left_spliter['feature_idx'],
                                left_spliter['threshold'],
                                node.depth+1,
                                is_leaf_left)
            self._tree_grow(node.left, X_left,y_left)


            right_spliter, score_right = self._best_splitter(X_right, y_right)
            is_leaf_right = is_leaf or len(np.unique(y_right)) == 1 or score_right==0

            node.right = TreeNode(right_spliter['feature_idx'],
                                right_spliter['threshold'],
                                node.depth+1,
                                is_leaf_right)
            self._tree_grow(node.right, X_right, y_right)

    def fit(self, X, y):
        self.depth = 1
        splitter, score = self._best_splitter(X,y)
        self.Tree = TreeNode(splitter['feature_idx'],
                            splitter['threshold'],
                             1,
                            False)
        self._tree_grow(self.Tree, X,y)

    def _predict_i(self,x):
        return self.Tree.predict(x)

    def predict(self,X):
        return [self._predict_i(x) for x in X]





class RegressionNode():
    """
    class Tree node
    Args:
        :feature_idx: spliter index
        :threshold: spliter threshold
        :depth: depth of the node in tree
        :is_leaf: bool, if is a leaf node

    method:
        activate:
            activate the node
            args:
                :X:
                :y:

        predict:
            predict an instance
            args:
                :x:
    """
    def __init__(self, feature_idx=None, threshold=None, depth=None,is_leaf=False):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.depth = depth
        self.left = None
        self.right = None

    def activate(self,X,y):

        if self.is_leaf:
            self.values = y
            return (None, None, None, None)

        X_left = X[X[:,self.feature_idx]< self.threshold]
        y_left = y[X[:,self.feature_idx]< self.threshold]
        X_right = X[X[:,self.feature_idx] >= self.threshold]
        y_right = y[X[:,self.feature_idx] >= self.threshold]

        return X_left, y_left, X_right, y_right

    def predict(self,x):
        if self.is_leaf:
            return self.values.mean()
        else:
            if x[self.feature_idx] <= self.threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)


class RegressionTree():
    """
    class Regression Tree

    Args:
        :criterion:
        :spittter:
        :max_depth:
        :min_sample_split:
        :min_samples_leaf:
        :max_features:
        :random_state:
        :max_leaf_nodes:
        :min_impurity_decrease:
    """
    def __init__(self, criterion='mse', splitter= 'exact', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None, max_leaf_nodes=None, min_score_decrease=0.0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_sample_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_score_decrease = min_score_decrease

    def _calculate_loss_decrease(self, y, y_left, y_right):
        if self.criterion == 'mse':
            loss = MeanSquareError()
        gain = loss.loss(y, np.mean(y))\
                -loss.loss(y_left, np.mean(y_left))*len(y_left)/len(y)\
                -loss.loss(y_right, np.mean(y_right))*len(y_right)/len(y)

        return gain

    def _best_splitter(self,X,y):
        score = 0
        splitter = dict(feature_idx=None,
                        threshold=None)
        if self.splitter == 'exact':
            proposal = exact_split_search(X,y)
        elif self.splitter == 'quantile':
            proposal =  quantile_split_search(X,y,random_state=self.random_state)

        for feature_idx, threshold in proposal:

            score_new = self._calculate_loss_decrease(y,
                                                 y[X[:,feature_idx]< threshold],
                                                 y[X[:,feature_idx]>= threshold])
            if score < score_new:
                score = score_new
                splitter = dict(feature_idx=feature_idx,
                                threshold=threshold)

        return splitter, score

    def _tree_grow(self,node,X,y):

        X_left, y_left, X_right, y_right =  node.activate(X,y)

        if not node.is_leaf:
            if node.depth == self.max_depth:
                is_leaf = True

            else:
                is_leaf = False

            left_spliter, score_left = self._best_splitter(X_left,y_left)
            is_leaf_left = is_leaf or len(np.unique(y_left)) == 1 or score_left ==0

            node.left = RegressionNode(left_spliter['feature_idx'],
                                left_spliter['threshold'],
                                node.depth+1,
                                is_leaf_left)
            self._tree_grow(node.left, X_left,y_left)


            right_spliter, score_right = self._best_splitter(X_right, y_right)
            is_leaf_right = is_leaf or len(np.unique(y_right)) == 1 or score_right==0

            node.right = RegressionNode(right_spliter['feature_idx'],
                                right_spliter['threshold'],
                                node.depth+1,
                                is_leaf_right)
            self._tree_grow(node.right, X_right, y_right)

    def fit(self, X, y):
        self.depth = 1
        splitter, score = self._best_splitter(X,y)
        self.Tree = RegressionNode(splitter['feature_idx'],
                            splitter['threshold'],
                             1,
                            False)
        self._tree_grow(self.Tree, X,y)

    def _predict_i(self,x):
        return self.Tree.predict(x)

    def predict(self,X):
        return [self._predict_i(x) for x in X]
