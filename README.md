# Machine Learning From Scratch

This a practice for ML algorithms 

## Tree 

### Decision Tree

```python
from Trees import DecisionTree
tree = DecisionTree.DecisionTree()
tree.fit(X,y)
tree.predict(X)
```
#### Decisoin Tree Result
<img src="figures/tree_clf result.png" />

### Regression Tree

```python
from Trees import DecisionTree
tree = DecisionTree.RegressionTree()
tree.fit(X,y)
tree.predict(X)
```
#### Regression Tree Result
<img src="figures/tree_reg depth1 rst.png" />
<img src="figures/tree_reg depth6 rst.png" />
<img src="figures/tree_reg depth16 rst.png" />

## Ensembles

```python
from Trees.Ensamble import Ensamble
rf = Ensamble([DecisionTree.DecisionTree(max_depth=3, splitter='quantile'),
              DecisionTree.DecisionTree(max_depth=3, splitter='quantile'),
              DecisionTree.DecisionTree(max_depth=3, splitter='quantile')])
             
```

### Random Forest

```python
from Trees.Ensamble import Ensamble, RandomForest
rf = RandomForest(max_features=.05,n_estimator=30,max_depth = 3)
rf.fit(X,y)
```

## GBM

### GBDT

```python
from Trees.GradientBoost import GradientBoostTree
gbdt = GradientBoostTree(n_estimator=n_estimator,
                         learning_rate=5e1,max_depth=depth)
gbdt.fit(X,y)
```
#### GBDT Results
<img src="figures/gbm 1 rst.png" />
<img src="figures/gbm 8 rst.png" />
