# *********************
# BGM xgb lgb cat
from Trees.DecisionTree import RegressionTree
import numpy as np
from utils.Loss import Entropy, Gini, MeanSquareError


class GradientBoostTree():
    """
    class gbm
    Args:

    """
    def __init__(self,n_estimator=5, learning_rate= .1, max_depth = 3, max_features = .5):
        self.n_estimator = n_estimator
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.models = []
        self.loss = MeanSquareError()

    def _boosting(self,X,y,preds):
        if len(self.models) < self.n_estimator:
            self.models.append(RegressionTree(max_depth = self.max_depth, max_features = self.max_features))
            gradient = np.array([-self.loss.gradient(y[i], preds[i]) for i in range(len(y))])
            self.models[-1].fit(X, gradient)
            preds += self.models[-1].predict(X) * self.learning_rate
            self._boosting(X,y,preds)

    def fit(self, X, y):
        self.models.append(RegressionTree(max_depth = self.max_depth, max_features = self.max_features))
        self.models[0].fit(X, np.full(len(y), np.mean(y)))
        preds = self.models[0].predict(X)
        self._boosting(X,y,preds)

    def predict(self,X):
        preds = self.models[0].predict(X)
        for model in self.models[1:]:
            preds += self.learning_rate*model.predict(X)

        return preds
