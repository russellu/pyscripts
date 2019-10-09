from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize

cancer = load_breast_cancer()
normdata = cancer.data #normalize(cancer.data,axis=0)
pca = PCA(n_components=5)
pca.fit(normdata)

transformed = pca.transform(normdata)

clf = GaussianNB()
clf.fit(cancer.data[0:350,:], cancer.target[0:350])  
classes = clf.predict(cancer.data[350:,:])
err = np.sum(np.abs(classes - cancer.target[350:]))/219

clf.fit(transformed[0:350,:], cancer.target[0:350])  
classes_pca = clf.predict(transformed[350:])
errpca = np.sum(np.abs(classes_pca - cancer.target[350:]))/219

print('errpca = ' + str(errpca))


