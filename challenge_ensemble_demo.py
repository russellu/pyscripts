import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('C:/shared/flinks/dataset.pkl')
ylabel = np.zeros(10000)
x_data = np.zeros([10000,6])

for cust in np.arange(0,10000):
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate

clf = DecisionTreeClassifier()

cvs = cross_validate(clf, x_data,ylabel,cv=10)
test_scores = cvs['test_score']

# get even split
nondefs = np.where(ylabel==0)[0]
defs = np.where(ylabel==1)[0]
balanced_xdata= np.zeros([2700*2,x_data.shape[1]])
balanced_xdata[0:2700,:] = x_data[nondefs[0:2700],:]
balanced_xdata[2700:,:] = x_data[defs,:]
balanced_ylabel = np.zeros(2700*2)
balanced_ylabel[0:2700] = 0
balanced_ylabel[2700:] = 1


cvs = cross_validate(clf, balanced_xdata,balanced_ylabel,cv=10)
test_scores = cvs['test_score']

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
cvs = cross_validate(clf, balanced_xdata,balanced_ylabel,cv=10)
test_scores = cvs['test_score']

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(balanced_xdata,balanced_ylabel)
cvs = cross_validate(clf, balanced_xdata,balanced_ylabel,cv=10)
test_scores = cvs['test_score']













"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_pickle('C:/shared/flinks/dataset.pkl')

ylabel = np.zeros(10000)
x_data = np.zeros([10000,6])

for cust in np.arange(0,10000):
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize

clf = DecisionTreeClassifier()
cvs = cross_validate(clf,x_data,ylabel,cv=10)
test_scores = cvs['test_score']

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

rf = RandomForestClassifier(n_estimators=10,criterion='gini')

rf_cvs = cross_validate(rf,x_data,ylabel,cv=10)
rf_test_scores = rf_cvs['test_score']

# get even split

nondefs = np.where(ylabel==0)[0]
defs = np.where(ylabel==1)[0]

balanced_xdata = np.zeros([2700*2,x_data.shape[1]])
balanced_xdata[0:2700,:] = x_data[defs,:]
balanced_xdata[2700:2700*2,:] = x_data[nondefs[0:2700],:]
balanced_ylabel = np.zeros(2700*2)
balanced_ylabel[0:2700] = 1

rf = GradientBoostingClassifier()
rf.fit(balanced_xdata,balanced_ylabel)
rf_cvs = cross_validate(rf,balanced_xdata,balanced_ylabel,cv=10)
rf_test_scores = rf_cvs['test_score']

print(np.mean(rf_test_scores))

"""















