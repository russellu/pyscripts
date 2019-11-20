import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('C:/shared/flinks/dataset_labeled.pkl')
unl_data = pd.read_pickle('C:/shared/flinks/dataset_unlabeled.pkl')
cat = pd.concat([data,unl_data],ignore_index=True)

ylabel = np.zeros(10000+5000+15822)
x_data = np.zeros([10000+5000+15822,6])

for cust in np.arange(0,30822):
    ts = cat[2][cust][1]
    
    if cust<=14999:
        isdef = cat[6][cust][1]
    else:
        isdef = -1
        
    cumsum_ts = np.cumsum(ts)
    loan_amt = cat[4][cust][1]
    
    ylabel[cust] = isdef
    
    x_data[cust,0] = np.mean(ts[np.where(ts>0)])
    x_data[cust,1] = np.mean(ts[np.where(ts<0)])
    x_data[cust,2] = cumsum_ts[-1] - cumsum_ts[0]
    x_data[cust,3] = np.max(ts)
    x_data[cust,4] = np.min(ts)
    x_data[cust,5] = loan_amt

x_data[np.isnan(x_data)] = 0

defs = np.where(ylabel[0:10000]==1)[0]
nondefs = np.where(ylabel[0:10000]==0)[0]
eqinds = np.hstack((defs,nondefs[0:2700]))
unl_eqinds = np.hstack((np.hstack((defs,nondefs[0:2700])),np.arange(15000,30822)))

from sklearn.semi_supervised import LabelSpreading

labprop = LabelSpreading(kernel='knn',n_neighbors=3)
labprop.fit(x_data[unl_eqinds,:],ylabel[unl_eqinds])
preds = labprop.predict(x_data)
plt.plot(preds)

print(np.sum(np.abs(preds[0:10000]-ylabel[0:10000]))/10000)

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()
clf.fit(x_data[eqinds,:],ylabel[eqinds])
preds2 = clf.predict(x_data[10000:15000])
preds2_proba = clf.predict_proba(x_data[10000:15000])
print('base error = ' + str(np.sum(np.abs(preds2[0:5000]-ylabel[10000:15000]))/5000))

from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(ylabel[10000:15000],preds2_proba[:,1])
area = auc(fpr,tpr)
print(area)


newpreds = preds[unl_eqinds]
newpreds[0:5400] = ylabel[eqinds]

clf = GradientBoostingClassifier()
clf.fit(x_data[unl_eqinds,:],newpreds)
preds3 = clf.predict(x_data[10000:15000,:])
preds3_proba = clf.predict_proba(x_data[10000:15000,:])
print('semi-sup error = ' + str(np.sum(np.abs(preds3[0:5000]-ylabel[10000:15000]))/5000))
fpr,tpr,thresholds = roc_curve(ylabel[10000:15000],preds3_proba[:,1])
area = auc(fpr,tpr)
print(area)














