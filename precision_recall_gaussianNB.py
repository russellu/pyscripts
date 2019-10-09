import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt 

gnb = GaussianNB()
cancer = load_breast_cancer()
y_prob = gnb.fit(cancer.data, cancer.target).predict_proba(cancer.data)

precision, recall, _ = precision_recall_curve(cancer.target,y_prob[:,1])

plt.plot(recall,precision)
plt.xlabel('Recall')
plt.ylabel('Precision')


# recall  = number of relevant documents retrieved / total number of relevant documents
# precision = number of relevant documents retrieved / total number of documents retrieved