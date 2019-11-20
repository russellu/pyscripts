import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from keras import optimizers

csv = pd.read_csv('C:/shared/dukascopy/EURUSD_5 Mins_Bid_2009.10.25_2019.10.31.csv')
# lstm expects input data to have shape [samples, time steps, features]
n_bars = 200
rawdata = np.zeros([csv.Close.shape[0],4])
rawdata[:,0] = csv.Open; 
rawdata[:,1] = csv.High; 
rawdata[:,2] = csv.Low;
rawdata[:,3] = csv.Close; 

x_data = np.zeros([int(csv.Close.shape[0]/n_bars),n_bars,4])
icount=0
for i in np.arange(0,rawdata.shape[0]-n_bars,n_bars):
    x_data[icount,:,:] = rawdata[i:i+n_bars,:] - rawdata[i,:]
    icount = icount + 1
 
y_data = np.zeros([int(csv.Close.shape[0]/n_bars),1])
for i in np.arange(1,x_data.shape[0]):
    y_data[i-1,0] = np.mean(x_data[i,0:5,3])


n_train_insts=int(x_data.shape[0]*0.8)
x_train = x_data[0:n_train_insts,:,:]
y_train = y_data[0:n_train_insts]
x_test = x_data[n_train_insts:,:,:]
y_test = y_data[n_train_insts:]

    
model = Sequential()
model.add(LSTM(100,input_shape=[n_bars,4]))
model.add(Dense(1,activation='tanh'))
model.compile(loss='mean_absolute_error',optimizer=optimizers.adam(lr=0.01))
model.fit(x_train, y_train, epochs=30,batch_size=300)

preds = model.predict(x_train)
plt.plot(y_train,preds,'.')
plt.title('corrcoef = ' + str(np.corrcoef(y_train.T,preds.T)[0,1]))

    













