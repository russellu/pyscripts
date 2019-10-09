from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import GaussianNB


cancer = load_breast_cancer()
pls2 = PLSRegression(n_components=2)
pls2.fit(cancer.data[0:350,:], cancer.target[0:350])
comps = pls2.transform(cancer.data)

# fit GNB to raw data
clf = GaussianNB()
clf.fit(cancer.data[0:350,:], cancer.target[0:350])  
classes = clf.predict(cancer.data[350:,:])
err = np.sum(np.abs(classes - cancer.target[350:]))/219

# fit GNB to PLS projected data
clf.fit(comps[0:350,:], cancer.target[0:350])  
classes_pls = clf.predict(comps[350:,:])
err_pls = np.sum(np.abs(classes_pls - cancer.target[350:]))/219

print("error using GNB on PLS projected data: " + str(err_pls))
print("error just using GNB: " + str(err))




