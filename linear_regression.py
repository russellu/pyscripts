import numpy as np 
from numpy.linalg import inv 

# initialize 3 instances, each with 3 attributes
instance1 = [1,4,1] 
instance2 = [4,1,3]
instance3 = [1,5,2]

# initialize the output (arbitarily)
Y = np.asarray([3,1,4])

# convert list to numpy array
X = np.asarray([instance1,instance2,instance3])

# perform least square regression using matrix multiplication
W = np.dot(np.dot(inv((np.dot(X.T,X))),X.T),Y)












