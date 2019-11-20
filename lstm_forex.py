import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from keras import optimizers

csv = pd.read_csv('C:/shared/dukascopy/EURUSD_15 Mins_Bid_2016.10.25_2019.10.31.csv')
# lstm expects input data to have shape [samples, time steps, features]

n_features = 1
rawdata = np.zeros([csv.Close.shape[0],n_features])
#rawdata[:,0] = csv.Open; 
#rawdata[:,1] = csv.High; 
#rawdata[:,2] = csv.Low;
rawdata[:,0] = csv.Close; 
n_bars = 25

x_data = np.zeros([int(csv.Close.shape[0]/n_bars),n_bars-1,n_features])
raw_x_data = np.zeros([int(csv.Close.shape[0]/n_bars),n_bars-1,n_features])
y_data = np.zeros([int(csv.Close.shape[0]/n_bars),1])

icount=0
for i in np.arange(0,rawdata.shape[0]-n_bars,n_bars):
    epoch_i = rawdata[i:i+n_bars,:]
    #x_data[icount,:,:] = (epoch_i - np.min(epoch_i))/ (np.max(epoch_i)-np.min(epoch_i) + 1)
    x_data[icount,:,:] = np.cumsum(np.diff(epoch_i,axis=0),axis=0)

    raw_x_data[icount,:,:] = np.diff(epoch_i,axis=0)
    icount = icount + 1



for i in np.arange(1,x_data.shape[0]):
    y_data[i-1,0] = np.mean(raw_x_data[i,0:2,n_features-1])


n_train_insts = int(x_data.shape[0]*0.7)
x_train = x_data[0:n_train_insts,:,:]
y_train = y_data[0:n_train_insts,0]
x_test = x_data[n_train_insts:,:,:]
y_test = y_data[n_train_insts:,0]

model = Sequential()
model.add(LSTM(25,input_shape=[n_bars-1,n_features],return_sequences=True))
model.add(LSTM(25,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(25))
model.add(Dropout(0.3))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer=optimizers.adam(lr=0.015))
model.fit(x_train, y_train, epochs=30,batch_size=80)

preds = model.predict(x_test)
plt.plot(preds,y_test,'o'); 
plt.title(np.corrcoef(preds.T,y_test.T)[0,1])

inds = np.argsort(preds,axis=0)



    
    













