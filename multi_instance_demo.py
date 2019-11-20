import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('C:/shared/flinks/dataset_labeled.pkl')
ylabel = np.zeros(15000)
all_transactions = np.zeros([4122934,1])
transaction_labs = np.zeros(4122934)
transaction_inds = []

n_transactions = 0 
for cust in np.arange(0,15000):
    ts = data[2][cust][1]
    isdef = data[6][cust][1]
    cumsum_ts = np.cumsum(ts)
    loan_amt = data[4][cust][1]
    
    ylabel[cust] = isdef
    all_transactions[n_transactions:n_transactions+ts.shape[0]] = ts
    transaction_inds.append(np.arange(n_transactions,n_transactions+ts.shape[0]))
    transaction_labs[n_transactions:n_transactions+ts.shape[0]] = isdef
    n_transactions = n_transactions + ts.shape[0]
    

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

n_clusts = 50
kmeans = MiniBatchKMeans(n_clusters=n_clusts).fit(all_transactions)
labs = kmeans.labels_
kfeatures = np.zeros([15000,n_clusts])
for i in np.arange(0,15000):
    kvals_i = labs[transaction_inds[i]]
    for j in np.arange(0,n_clusts):
        kfeatures[i,j] = np.sum(kvals_i==j)
    

defs = np.where(ylabel[0:10000]==1)[0]
nondefs = np.where(ylabel[0:10000]==0)[0]    
eqinds = np.hstack((defs,nondefs[0:2700]))
    
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=1000,max_depth=10)
clf.fit(kfeatures[eqinds,:],ylabel[eqinds])
preds = clf.predict_proba(kfeatures[10000:15000,:])

from sklearn.metrics import roc_curve,auc

fpr,tpr,_ = roc_curve(ylabel[10000:15000],preds[:,1])
print(auc(fpr,tpr))














