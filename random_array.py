import numpy as np
import matplotlib.pyplot as plt


randarr = np.random.random((1000,50))

r = np.sum(xdata[:,1]*ydata[:,0]) \
    /np.sqrt(np.sum(xdata[:,1]*xdata[:,1]) \
    *np.sum(ydata*ydata))

