import numpy as np 
from numpy.linalg import inv 
import matplotlib.pyplot as plt 

xdata = np.zeros([10,2])
xdata[:,0] = 1
xdata[:,1] = np.arange(0,10)
ydata = xdata[:,1] + np.random.rand(10)*20
#plt.plot(xdata[:,1],ydata);
plt.plot(xdata[:,1],ydata,'o')

#xdata = np.expand_dims(xdata,axis=1)
ydata = np.expand_dims(ydata,axis=1)

# compute the weights
W = np.dot(np.dot(inv((np.dot(xdata.T,xdata))),xdata.T),ydata)

# plot the least square line 
plt.plot(xdata[:,1],np.dot(xdata,W)); plt.plot(xdata[:,1],ydata,'o')

# subtract mean from ydata and xdata
xdata[:,1] = xdata[:,1] - np.mean(xdata[:,1])
ydata = ydata - np.mean(ydata)

# get the correlation coefficient 
r = np.sum(xdata[:,1]*ydata[:,0]) \
    /np.sqrt(np.sum(xdata[:,1]*xdata[:,1]) \
    *np.sum(ydata*ydata))
