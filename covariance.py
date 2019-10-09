import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()

covmat = np.cov(iris.data.T)
fig = plt.figure()
plt.plot(iris.data[:,0],iris.data[:,2],'o')
plt.xlabel('sepal length'); plt.ylabel('petal length');
plt.title('covariance = ' + str(np.round((covmat[0,2]*100))/100))

fig = plt.figure()
plt.plot(iris.data,'o')
plt.xlabel('instance'); plt.ylabel('attribute value')

fig = plt.figure(); 
cs = plt.imshow(covmat); fig.colorbar(cs)
plt.xticks([0,1,2,3],['spl.L','spl.W','ptl.L','ptl.W'])
plt.yticks([0,1,2,3],['spl.L','spl.W','ptl.L','ptl.W'])
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(iris.data)
transformed = pca.transform(iris.data)
pca_covmat = np.cov(transformed.T)

fig = plt.figure(); 
cs = plt.imshow(pca_covmat); fig.colorbar(cs)
plt.xticks([0,1,2,3],['comp1','comp2','comp3','comp4'])
plt.yticks([0,1,2,3],['comp1','comp2','comp3','comp4'])
plt.title('PCA component covariance matrix')
plt.show()

white_cov = np.zeros([4,4]);
white_cov[0,0]=1;white_cov[1,1]=1;white_cov[2,2]=1;white_cov[3,3]=1;
fig = plt.figure(); 
cs = plt.imshow(white_cov); fig.colorbar(cs)
plt.xticks([0,1,2,3],['comp1','comp2','comp3','comp4'])
plt.yticks([0,1,2,3],['comp1','comp2','comp3','comp4'])
plt.title('covariance matrix of whitened data')
plt.show()







