import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt 
from numpy.linalg import inv 

iris = load_iris()

# set class 1 (setosa) output to 1, zero for all others
setosa_output = np.zeros([150,1])
setosa_output[0:50,:] = 1; 
# set class 2 (versicolor) output to 1, zero for all others
versicolor_output = np.zeros([150,1])
versicolor_output[50:100,:] = 1; 
# set class 3 (virginica) output to 1, zero for all others
virginica_output = np.zeros([150,1])
virginica_output[100:150,:] = 1; 

# perform Logistic Regression 
W_setosa = np.dot(np.dot(inv((np.dot(iris.data.T,iris.data))),      \
                         iris.data.T),setosa_output)
W_versicolor = np.dot(np.dot(inv((np.dot(iris.data.T,iris.data))),  \
                             iris.data.T),versicolor_output)
W_virginica = np.dot(np.dot(inv((np.dot(iris.data.T,iris.data))),   \
                            iris.data.T),virginica_output)

# generate predictions:
pred_setosa = np.dot(iris.data,W_setosa)
pred_versicolor = np.dot(iris.data,W_versicolor)
pred_virginica = np.dot(iris.data,W_virginica)

# get the predicted class
pred_all = np.zeros([150,3]) # create empty array to store predictions
pred_all[:,0] = pred_setosa[:,0]
pred_all[:,1] = pred_versicolor[:,0]
pred_all[:,2] = pred_virginica[:,0] 

max_inds = np.argmax(pred_all,axis=1)

accuracy_setosa = np.sum(max_inds[0:50]==0)/50 
accuracy_versicolor = np.sum(max_inds[50:100]==1)/50 
accuracy_virginica = np.sum(max_inds[100:150]==2)/50 



