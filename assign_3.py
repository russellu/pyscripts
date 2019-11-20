# assignment 3 - sklearn + validation techniques

# goal 1: become familiar with a variety of classifiers on different datasets in sklearn
# goal 2: gain experience with validation techniques (cross-validation, ROC curve, t-test)
# the assignment mainly just involves calling sklearn functions and displaying results

# DUE: Wednesday November 20th, 2019, 15% of final grade
# please hand in your code as well as a pdf with all relevant figures
# Prof. R.Butler, Johnson 114A, Office Hours MWF 9:00-11:00am


# ----------------------------------------------------------------------------
# QUESTION 1
# import and load the toy datasets from sklearn

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer 

boston = load_boston()              # regression
iris = load_iris()                  # classification 
diabetes = load_diabetes()          # regression
digits = load_digits()              # classification 
linnerud = load_linnerud()          # regression
wine = load_wine()                  # classification
cancer = load_breast_cancer()       # classification

# QUESTION 1A 
# import and initialize the following classifers from sklearn: 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

gnb = GaussianNB() # probabilistic
tree = tree.DecisionTreeClassifier() # divide and conquer
neigh = KNeighborsClassifier() # instance based
mlp = MLPClassifier() # neural network


# train and run each classifier on each classification dataset separately,  
# resulting in a 4x4 matrix (dataset rows, classifier columns) showing the accuracy of
# 4 classifiers above on each classification dataset (just use entire dataset, no test/train). 
# classification datasets: iris, digits, wine, breast cancer
# display this matrix using imshow from matplotlib, 
# be sure to show proper x and y tick labels and colorbar

# using the 4x4 matrix above, answer the following: 

# which classifier has highest mean accuracy across all datasets?
# which dataset has the highest mean accuracy across all classifiers?


# QUESTION 1B
# import and initialize the following regression algorithms from sklearn:
from sklearn import linear_model
from sklearn import svm

reg = linear_model.LinearRegression() # linear regression
svr = svm.SVR() # support vector regression

# create a 3x2 matrix (dataset rows, classifier columns) showing performance of 2 
# regression techniques above on the three regression datasets
# (for linnerud, use only chinups)
# which regression technique has lower mean-squared error (across datasets)
# which dataset has lowest mean squared-error (across regression methods)


# ----------------------------------------------------------------------------
# QUESTION 2 california housing predictions + validation

from sklearn.datasets import fetch_california_housing
# fetch california housing dataset
cali = fetch_california_housing()

# QUESTION 2A
# using gaussian naive bayes:
# for each instance output a probability that the house is worth over $300k
# (target variable is in units of $100,000's)

# QUESTION 2B perform k-fold (k=10) cross-validation (CV):
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, data, target, cv=10)

# answer the following: 
# what is the average error across all folds on the test set? (using GNB)
# how does this compare to the resubstitution error (error on training data)?

#QUESTION 2C Plot the ROC curve using the training data from 2a 
# this is also known as the 'resubstitution ROC curve' (ROC from training data)
# (ie, train GNB on all the instances, then plot ROC curve using probabilities 
# from predictions on instances that GNB was trained on)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y,score) 

# answer the following: 
# what is the area under the ROC curve ? 
# can you increase the area under ROC by denoising or removing attributes, 
# of performing some other type of transformation (PCA, PLS, etc.)? 

# ----------------------------------------------------------------------------
# QUESTION 3  (using california housing dataset from previous question)
# using the following 3 classifiers:
# i) Gaussian Naive Bayes (same as above)
# ii) k-nearest neighbour with k=3 (yes, it can output probabilities)
# iii) random forest (also outputs probabilities)

# Question 3A:
# plot the average ROC curve with error bars over 10 tests sets from 10-fold CV 
# (ie, plot one ROC curve for each fold's test set predictions, and average them 
# all to create a single, smoother ROC curve)
# error bars should show standard deviation across different folds

# answer the following:

# which algorithm gives the highest area under average ROC curve from above? 
# using a t-test, compare the algorithm with the higest performance to the 
# other two algorithms, to output a p-value and state if the p-value is 
# significant (p<0.05)











