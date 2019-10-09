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

data = pd.read_csv("/media/sf_shared/dukas/GBPUSD_Hourly_Bid_2013.11.03_2018.11.09.csv",
                   sep=" |,",
                   header=None,
                   engine='python')

vals = data.values[1:,2:]
x = np.array(list(vals[:,:]),dtype=np.float)
close_data = x[:,3] 
# create moving average
def mvg(data,mvg_period):
    mvg = np.zeros(data.shape[0])
    for i in np.arange(mvg_period,data.shape[0]):
        if i==mvg_period:
            mvg[i] = np.mean(data[0:i])
        else:
            mvg[i] = mvg[i-1] - data[i-mvg_period]/mvg_period + data[i]/mvg_period
    return mvg
# create bollinger band

mvg_period = 25
std_mul = 3.5
close_mvg = mvg(close_data,mvg_period)
bbands = np.zeros((2,close_data.shape[0]))
for i in np.arange(mvg_period, close_data.shape[0]):
    std_i = np.std(close_data[i-mvg_period:i])
    bbands[0,i] = close_mvg[i] + std_i*std_mul
    bbands[1,i] = close_mvg[i] - std_i*std_mul

xvals = np.zeros((2,close_data.shape[0]))
xvals[0,:] = bbands[0,:] - close_data
xvals[1,:] = bbands[1,:] - close_data
yvals = np.zeros((close_data.shape))


hist_pts = 2
x_train = np.zeros((hist_pts*2,close_data.shape[0]-1-hist_pts))
y_train = np.zeros(close_data.shape[0]-1-hist_pts)
icount = 0
for i in np.arange(mvg_period+hist_pts,close_data.shape[0]-1):
    x_i = xvals[:,i-hist_pts:i].flatten()
    x_train[:,icount] = x_i
    y_train[icount] = close_data[i+1] - close_data[i]
    icount = icount + 1
    
# create bollinger bands
RESHAPED = hist_pts*2
OPTIMIZER = Adam(lr=0.0005)

model = Sequential()
model.add(Dense(500, input_shape=(RESHAPED,)))
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(optimizer=OPTIMIZER, loss='mean_absolute_error')

historY = model.fit(x_train.transpose(), y_train.transpose(),
                    batch_size=300, epochs=150, validation_split=0.2)

predicted = model.predict(x_train.transpose())

sorts = np.argsort(predicted,axis=0)
sorted_y = y_train[sorts]

print(np.mean(sorted_y[0:150]))
print(np.mean(sorted_y[-1-150:-1]))











