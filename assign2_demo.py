import numpy as np
import matplotlib.pyplot as plt

n_elems=200
raw = np.random.rand(1000,n_elems)

msub_raw = raw - np.tile(np.expand_dims(np.mean(raw,axis=1),axis=1),[1,n_elems])

corrmat = np.zeros([1000,1000])

import time as time

t1 = time.time()
for i in np.arange(0,1000):
    tilevec_i = np.tile(np.expand_dims(msub_raw[i,:],axis=0),[1000,1])
    corrmat[i,:] = np.sum(tilevec_i * msub_raw,axis=1)  \
        / (np.sqrt(np.sum(tilevec_i*tilevec_i,axis=1))* np.sqrt(np.sum(msub_raw*msub_raw,axis=1)))
t2 = time.time()
print(t2-t1)

triangle = np.tril(corrmat)
vals = triangle[np.where(np.logical_and(triangle != 0, triangle < 0.999))]

hist = np.histogram(vals,100)

plt.plot(hist[1][1:],hist[0])

































































"""
assignment 2 question 1 demo
random vector initialization
mean subtraction
tiling
correlation coefficient
histogram and x-axis 
effects of sample size on probability of high corrcoeff
"""
"""
import numpy as np
import matplotlib.pyplot as plt 
n_elems=10
raw = np.random.rand(1000,n_elems)
# mean subtract data
msub_raw = raw - np.tile(np.expand_dims(np.mean(raw,axis=1),axis=1),[1,n_elems]) 
corrmat = np.zeros([1000,1000])
import time
t1 = time.time()
for i in np.arange(0,1000):
    tilevec_i = np.tile(np.expand_dims(msub_raw[i,:],axis=0),[1000,1])
    corrmat[i,:] = np.sum(tilevec_i*msub_raw,axis=1)        \
    /   (np.sqrt(np.sum(tilevec_i*tilevec_i,axis=1))*np.sqrt(np.sum(msub_raw*msub_raw,axis=1)))
t2 = time.time()
t = t2-t1;
tril = np.tril(corrmat)
vals = tril[np.where(np.logical_and(tril!=0,  tril<0.999))]
hist = np.histogram(vals,100)
#plt.plot(hist[1][1:],hist[0]); plt.xlabel('bin value'); plt.ylabel('count')

nonzcorrmat = corrmat; nonzcorrmat[nonzcorrmat>0.99]=0
corr_inds = np.where(nonzcorrmat==np.max(nonzcorrmat))
plt.figure
plt.plot(raw[corr_inds[0][0],:],raw[corr_inds[0][1],:],'o')

"""








