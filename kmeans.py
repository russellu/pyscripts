import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import imageio as imageio
from sklearn.cluster import KMeans 

mars_bar = imageio.imread('/media/sf_shared/marsbar.png')

reshape_mbar = np.reshape(mars_bar,[200*508,3])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(reshape_mbar[0::50,0], reshape_mbar[0::50,1], reshape_mbar[0::50,2])
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0).fit(reshape_mbar)

labs0 = np.where(kmeans.labels_==0)[0]
labs1 = np.where(kmeans.labels_==1)[0]
labs2 = np.where(kmeans.labels_==2)[0]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(reshape_mbar[labs0[0::50],0], reshape_mbar[labs0[0::50],1], reshape_mbar[labs0[0::50],2])
ax.scatter(reshape_mbar[labs1[0::50],0], reshape_mbar[labs1[0::50],1], reshape_mbar[labs1[0::50],2])
ax.scatter(reshape_mbar[labs2[0::50],0], reshape_mbar[labs2[0::50],1], reshape_mbar[labs2[0::50],2])
plt.show()

mbar_labels = np.reshape(kmeans.labels_,[200,508])
plt.imshow(mbar_labels)

from sklearn import tree
import pydotplus
from IPython.display import Image

clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(reshape_mbar, kmeans.labels_)

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=['red','green','blue'], 
                                class_names=['background', 'wrapper','label'],
                                rounded=True, filled=True) 

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


