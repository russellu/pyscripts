import numpy as np
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt 
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

data = load_iris().data 

tdata = np.zeros([100,4])
tdata[:,1:4] = data[0:100,0:3]
tdata[:,0] = 1
classes = np.zeros([100]); classes[50:100] = 1;

fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(tdata[0:50,0], tdata[0:50,1], tdata[0:50,2])
ax.scatter(tdata[50:100,0], tdata[50:100,1], tdata[50:100,2])
pyplot.show()

"""
perceptron learning algorithm:
    set all weights to zero
    until all instances in training set are correctly classified
        for each instance I in the training set
            if I is classified incorrectly by the perceptron
                if I belongs to the first class, add it to the weight vector
                else subtract I from weight vector
"""
# set all weights to zero
weights = np.zeros([4,1]) 

# until all instances in training set are correctly classified
for i in np.arange(0,100):
    for inst in np.arange(0,tdata.shape[0]): # for each instance 
        prediction = np.dot(tdata[inst,:],weights)
        if prediction>0: 
            predicted_class=1
        else:
            predicted_class=0
        if predicted_class != classes[inst]: # if inst is classed incorrectly
            if classes[inst]==1: # if inst belongs to first class, add to weight vec
                weights = weights + np.expand_dims(tdata[inst,:],1)
            else: # else subtract it from weight vec
                weights = weights - np.expand_dims(tdata[inst,:],1)
        
        
predictions = np.dot(tdata,weights)
plt.plot(predictions,'o'); plt.plot([0,100],[0,0])

"""
x = np.linspace(4.5,7.0,10)
y = np.linspace(0.9,1.05,10)
X,Y = np.meshgrid(x,y)
Z=-X*weights[1]-Y*weights[2]-weights[0]
Z = Z*weights[3]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(tdata[0:50,0], tdata[0:50,1], tdata[0:50,2])
ax.scatter(tdata[50:100,0], tdata[50:100,1], tdata[50:100,2])
surf = ax.plot_surface(X, Y, Z,alpha=0.2)
"""