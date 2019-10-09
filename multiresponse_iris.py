import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
from numpy.linalg import inv 


iris = load_iris()

petals = np.zeros([100,2])
petals[:,0] = 1; # augment with 1s so we can estimate intercept
petals[:,1] = iris.data[0:100,0] # place petal length only in petals array

setosa_y = np.zeros([100,1]); 
setosa_y[0:50,0] = 1

versicolor_y = np.zeros([100,1]); 
versicolor_y[50:100,0] = 1

w_setosa =  np.dot(np.dot(inv((np.dot(petals.T,petals))),      \
                          petals.T),setosa_y)

w_versicolor =  np.dot(np.dot(inv((np.dot(petals.T,petals))),      \
                          petals.T),versicolor_y)


pred_setosa = np.dot(petals,w_setosa)
pred_versicolor = np.dot(petals,w_versicolor)

plt.plot(petals[:,1],versicolor_y,'o'); 
plt.plot(petals[:,1],np.dot(petals,w_versicolor))
plt.xlabel('sepal length')







