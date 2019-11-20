import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('C:/shared/flinks/dataset.pkl')

ylabel = np.zeros(10000)
x_data = np.zeros([10000,3])


for cust in np.arange(0,10000):
    ts = data[2][cust][1]
    isdef = data[6][cust][1]
    cumsum_ts = np.cumsum(ts)
    
    ylabel[cust] = isdef
    
    x_data[cust,0] = np.mean(ts[np.where(ts>0)])
    x_data[cust,1] = np.mean(ts[np.where(ts<0)])
    x_data[cust,2] = cumsum_ts[-1] - cumsum_ts[0]
    

x_data[np.isnan(x_data)] = 0

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(1)
clf.fit(x_data,ylabel)

preds = clf.predict_proba(x_data)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr,tpr,thresholds = roc_curve(ylabel,preds[:,1])
roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
plt.title(roc_auc)

