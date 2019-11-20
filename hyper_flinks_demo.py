import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('C:/shared/flinks/dataset_labeled.pkl')
ylabel = np.zeros(15000)
x_data = np.zeros([15000,6])

for cust in np.arange(0,15000):
    ts = data[2][cust][1]
    isdef = data[6][cust][1]
    cumsum_ts = np.cumsum(ts)
    loan_amt = data[4][cust][1]
    
    ylabel[cust] = isdef
    
    x_data[cust,0] = np.mean(ts[np.where(ts>0)])
    x_data[cust,1] = np.mean(ts[np.where(ts<0)])
    x_data[cust,2] = cumsum_ts[-1] - cumsum_ts[0]
    x_data[cust,3] = np.max(ts)
    x_data[cust,4] = np.min(ts)
    x_data[cust,5] = loan_amt

x_data[np.isnan(x_data)] = 0

from sklearn.utils import resample 


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

n_ests = np.arange(50,350,50)
depths = np.arange(1,6,1)
hypmat = np.zeros([6,5])

for i in np.arange(0,n_ests.shape[0]):
    for j in np.arange(0,depths.shape[0]):
        def_inds = np.where(ylabel[0:10000]==1)[0]
        nondef_inds = resample(np.where(ylabel[0:10000]==0)[0],n_samples=2700,replace=False)
        all_inds = np.hstack((def_inds,nondef_inds))
        clf = GradientBoostingClassifier(n_estimators=n_ests[i],max_depth=depths[j])
        scores = cross_validate(clf,x_data[all_inds,:],ylabel[all_inds],cv=10)
        hypmat[i,j] = np.mean(scores['test_score'])
        print('i='+str(i) + ' j='+str(j))
      



clf = GradientBoostingClassifier(n_estimators=n_ests[2],max_depth=depths[4])
clf.fit(x_data[all_inds,:],ylabel[all_inds])
probs = clf.predict_proba(x_data[10000:,:])
from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(ylabel[10000:],probs[:,1])
area = auc(fpr,tpr)
print(area)









