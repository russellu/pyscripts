import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
iris = load_iris()

y_pred = clf.fit(iris.data, iris.target).predict(iris.data)

scores = cross_val_score(clf, iris.data, iris.target, cv=10)



