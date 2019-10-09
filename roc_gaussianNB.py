import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 

gnb = GaussianNB()
cancer = load_breast_cancer()
y_prob = gnb.fit(cancer.data, cancer.target).predict_proba(cancer.data)
y_pred = gnb.fit(cancer.data, cancer.target).predict(cancer.data)

fpr,tpr,thresh = roc_curve(cancer.target,y_prob[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('area under curve = ' + str(roc_auc))

accuracy = 1-np.sum(np.abs(y_pred-cancer.target))/cancer.target.shape[0]

# true positive rate = (100*TP)/(TP+FN)
# false positive rate = (100*FP)/(FP+TN)


