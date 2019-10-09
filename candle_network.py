import matplotlib.pyplot as plt 
from keras.models import Sequential, Model
import numpy as np
from keras.datasets import mnist
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import regularizers
from scipy import signal
from scipy.interpolate import griddata
import matplotlib.animation as animation
from mne.preprocessing import ICA
import pandas as pd

data = pd.read_csv("/media/sf_shared/dukas/USDCAD_Hourly_Bid_2013.11.03_2018.11.09.csv",
                   sep=" |,",
                   header=None,
                   engine='python')

vals = data.values[1:,2:]
x = np.array(list(vals[:,:]),dtype=np.float)

def test_hypers(x, nxts=50, nyts=1):
    # put data in training format:
    print("testing x with " + str(nxts) + " nxts and " + str(nyts) + "nyts")
    full_xtrain = np.zeros((nxts, x.shape[0]-nxts*2))
    full_ytrain = np.zeros((nyts, x.shape[0]-nxts*2))
    for i in np.arange(0,x.shape[0]-nxts*2):
        full_xtrain[:,i] = x[i:i+nxts,3] - x[i+nxts,3]
        full_ytrain[:,i] = x[i+nxts+1:i+nxts+nyts+1,3] - x[i+nxts,3]     

    # train on every other time-point
    xtrain = full_xtrain[:,0::2]
    ytrain = full_ytrain[:,0::2]
    xtest = full_xtrain[:,1::2]
    ytest = full_ytrain[:,1::2]
    
    RESHAPED = nxts
    OPTIMIZER = Adam(lr=0.00005)
    
    model = Sequential()
    model.add(Dense(24, input_shape=(RESHAPED,)))#kernel_regularizer=regularizers.l1(0.00001)
    model.add(Activation('linear'))
    model.add(Dropout(0.3))
    model.add(Dense(nyts))
    model.add(Activation('linear'))
    
    model.compile(optimizer=OPTIMIZER, loss='mean_absolute_error')
    history_Y = model.fit(xtrain.transpose(), ytrain.transpose(), batch_size=300, epochs=200, validation_split=0.2)
    predicted = model.predict(xtest.transpose())
    
    return predicted, ytest, history_Y

results = [] 
hypers = np.arange(5,25,3)
for h in hypers:
    predicted,ytest,history_Y = test_hypers(x, nxts=h, nyts=1)     
    sorts = np.argsort(predicted[:,0])
    results.append(ytest[0,sorts])
    invsorts = np.flip(np.argsort(predicted[:,0])) 
    results.append(ytest[0,invsorts])
    

    
res2 = np.array(results)
sorts = res2[0::2]
invsorts = res2[1::2]

mdiffs = np.zeros([2,sorts.shape[0]])
for i in np.arange(0,sorts.shape[0]):
    mdiffs[0,i] = np.mean(sorts[i][0:2000])
    mdiffs[1,i] = np.mean(invsorts[i][0:2000])
    
    
    
#plt.plot(np.mean(ytest[:,sorts[0:500]],axis=1)); 
#plt.plot(np.mean(ytest[:,invsorts[0:500]],axis=1))


