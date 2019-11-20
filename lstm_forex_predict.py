import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from keras import optimizers

csv = pd.read_csv('C:/shared/dukascopy/EURUSD_Hourly_Ask_2009.10.25_2019.10.31.csv')
# lstm expects input data to have shape [samples, time steps, features]
n_bars = 15
rawdata = np.zeros([csv.Close.shape[0],1])
rawdata[:,0] = csv.Close; 

x_data = np.zeros([int(csv.Close.shape[0]/n_bars),n_bars,1])
icount=0
for i in np.arange(0,rawdata.shape[0]-n_bars,n_bars):
    x_data[icount,:,:] = np.cumsum(np.diff(rawdata[i:i+n_bars+1,:],axis=0),axis=0)
    icount = icount + 1

y_data = x_data[1:,0:5,0]
y_data = y_data + np.tile(x_data[0:-1,-1],[1,5])
x_data = x_data[0:-1,:,:]



n_trains=int(x_data.shape[0]*0.8)
x_train = x_data[0:n_trains,:,:]
y_train = y_data[0:n_trains,:]
x_test = x_data[n_trains:,:,:]
y_test = y_data[n_trains:,:]
    
model = Sequential()
model.add(LSTM(15,input_shape=[n_bars,1]))
model.add(Dropout(0.3))
model.add(Dense(5,activation='linear'))
model.compile(loss='mean_squared_error',optimizer=optimizers.Adagrad())
model.fit(x_train, y_train, epochs=200,batch_size=200)

preds = model.predict(x_test)

corrs = np.zeros([preds.shape[0]])
for i in np.arange(0,preds.shape[0]):
    corrs[i] = np.corrcoef(preds[i,:],y_test[i,:])[0,1]
    
plt.plot(preds[0:50,:].T)

153+259


    
    













