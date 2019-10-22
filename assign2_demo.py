import numpy as np
import matplotlib.pyplot as plt 

n_elems=10

raw = np.random.rand(1000,n_elems)

# mean subtract data
msub_raw = raw - np.tile(np.expand_dims(np.mean(raw,axis=1),axis=1),[1,n_elems]) 

corrmat = np.zeros([1000,1000])

for i in np.arange(0,1000):
    tilevec_i = np.tile(np.expand_dims(msub_raw[i,:],axis=0),[1000,1])
    corrmat[i,:] = np.sum(tilevec_i*msub_raw,axis=1)        \
    /   (np.sqrt(np.sum(tilevec_i*tilevec_i,axis=1))*np.sqrt(np.sum(msub_raw*msub_raw,axis=1)))

tril = np.tril(corrmat)

vals = tril[np.where(np.logical_and(tril!=0,  tril<0.999))]

hist = np.histogram(vals,100)

plt.plot(hist[1][1:],hist[0])










