from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs

n_samples = 250
center = [(0, 5)]
X,_ = make_blobs(n_samples=n_samples, n_features=2,centers=center,random_state=0)
theta = np.radians(60)
t = np.tan(theta)
shear_x = np.array(((0.15, t), (0, 0.15))).T
X_rotated = X.dot(shear_x)


pca = PCA(n_components=2)
pca.fit(X_rotated)
mu = np.mean(X_rotated,axis=0)
plt.plot(X_rotated[:, 0], X_rotated[:, 1],'o')
comps = pca.components_ 
plt.plot([mu[0]-comps[0,0],mu[0]+comps[0,0]],[mu[1]-comps[0,1],mu[1]+comps[0,1]],lw=10)
plt.plot([mu[0]-comps[1,0],mu[0]+comps[1,0]],[mu[1]-comps[1,1],mu[1]+comps[1,1]],lw=6)

transformed = pca.transform(X_rotated)
plt.plot(transformed[:,0],transformed[:,1],'o')
plt.xlabel('component 1'); plt.ylabel('component 2')

plt.plot(transformed[:,0],np.zeros([250]),'o'); 
plt.xlabel('component 1')



