# import stuff
# show type
# manipulate ndarrays
# visualize data (histogram, scatter plot)
# do simple clustering

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()

kmeans = KMeans(3).fit(iris.data)





