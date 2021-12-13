import numpy as np
import scipy.stats as stat
from tqdm import tqdm
from utils.resampling import Bootstrap
from Trees.DecisionTree import DecisionTree


class Ensamble():
    """
    Class Ensambel learning Object
        currently a bootstrap learning
        using utils.resampling.Bootstrap().__iter__()
    Args:
        models: list of estimators
        random_state:

    """

    def __init__(self, models,random_state):
        self.models = models
        self.n_estimator = len(self.models)
        self.random_state = random_state

    def fit(self, X, y):
        self.resample_generator = Bootstrap(X,y,self.n_estimator, self.random_state)
        for model, (Xi, yi) in tqdm(zip(self.models, self.resample_generator),total = self.n_estimator):
            model.fit(Xi, yi)

    def predict(self,X):
        """
        hard voting
        soft to be updated...
        """
        preds = [model.predict(X) for model in self.models]
        preds = stat.mode(preds,axis=0)[0][0]
        return preds



class RandomForest(Ensamble):
    def __init__(self, max_depth = 2, n_estimator = 3, max_features=0.5 ,random_state = None):
        self.max_depth = max_depth
        self.n_estimator = n_estimator
        self.max_features = max_features
        self.random_state = random_state
        self.models = [DecisionTree(max_depth = max_depth,
                                    max_features= max_features,
                                    splitter = 'quantile') for _ in range(n_estimator)]
        super().__init__(self.models, random_state)
