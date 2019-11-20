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

from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

nontest = np.hstack((np.arange(0,10000),np.arange(15000,30822)))

semisup_ylabel = ylabel.copy()
semisup_ylabel[10000:15000] = -1

labspread = LabelSpreading(kernel='knn',n_neighbors=3)
labspread.fit(x_data[:,:],semisup_ylabel[:])
preds = labspread.predict(x_data[:,:])
#plt.plot(preds)

base_clf = GradientBoostingClassifier()
base_clf.fit(x_data[0:10000,:],ylabel[0:10000])
base_probs = base_clf.predict_proba(x_data[10000:15000,:])
base_fp,base_tp,_ = roc_curve(ylabel[10000:15000],base_probs[:,1])
base_auc = auc(base_fp,base_tp)

newpreds = preds
newpreds[0:10000] = ylabel[0:10000]

semisup_clf = GradientBoostingClassifier()
semisup_clf.fit(x_data,newpreds)
semisup_probs = semisup_clf.predict_proba(x_data[10000:15000,:])
semisup_fp,semisup_tp,_ = roc_curve(ylabel[10000:15000],semisup_probs[:,1])
semisup_auc = auc(semisup_fp,semisup_tp)
print(semisup_auc)



























