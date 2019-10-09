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

#data = np.loadtxt("/media/sf_shared/dukas/AUDCAD_Hourly_Bid_2013.11.03_2018.11.09.csv",

data = pd.read_csv("/media/sf_shared/dukas/AUDCAD_Hourly_Bid_2013.11.03_2018.11.09.csv",
                   sep=" |,",
                   header=None,
                   engine='python')

vals = data.values[1:,2:]
x = np.array(list(vals[:,:]),dtype=np.float)

#predict volume 
nxts = 192
nyts = 24
xtrain = np.zeros((nxts,x.shape[0]-nxts*2))
ytrain = np.zeros((nyts,x.shape[0]-nxts*2))
for i in np.arange(0,x.shape[0]-nxts*2):
    xtrain[:,i] = x[i:i+nxts,4]
    ytrain[:,i] = x[i+nxts:i+nxts+nyts,4]
    

RESHAPED = nxts
OPTIMIZER = Adam(lr=0.00005)

model = Sequential()
model.add(Dense(36, input_shape=(RESHAPED,)))#kernel_regularizer=regularizers.l1(0.00001)
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(nyts))
model.add(Activation('linear'))

model.compile(optimizer=OPTIMIZER, loss='mean_absolute_error')
historY = model.fit(xtrain.transpose(), ytrain.transpose(), batch_size=300, epochs=200, validation_split=0.2)

predicted = model.predict(xtrain.transpose())

n=11000; plt.plot(predicted[n,:]) ; plt.plot(ytrain[:,n])

