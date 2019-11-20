import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from keras import optimizers

csv = pd.read_csv('C:/shared/dukascopy/GBPJPY_Hourly_Ask_2009.10.25_2019.10.31.csv')
# lstm expects input data to have shape [samples, time steps, features]

corrs = np.zeros([25,50])
n_count=0 
for n in np.arange(5,30,1):
    
    n_bars = n
    rawdata = np.zeros([csv.Close.shape[0],4])
    rawdata[:,0] = csv.Open; 
    rawdata[:,1] = csv.High; 
    rawdata[:,2] = csv.Low;
    rawdata[:,3] = csv.Close; 
    
    x_data = np.zeros([int(csv.Close.shape[0]/n_bars),n_bars-1,4])
    icount=0
    for i in np.arange(0,rawdata.shape[0]-n_bars,n_bars):
        x_data[icount,:,:] = np.diff(rawdata[i:i+n_bars,:],axis=0)
        icount = icount + 1
     
    y_data = np.zeros([int(csv.Close.shape[0]/n_bars),1])
    for i in np.arange(1,x_data.shape[0]):
        y_data[i-1,0] = np.mean(x_data[i,0:2,3])
    
    n_train_insts=int(x_data.shape[0]*0.8)
    x_train = x_data[0:n_train_insts,:,:]
    y_train = y_data[0:n_train_insts]
    x_test = x_data[n_train_insts:,:,:]
    y_test = y_data[n_train_insts:]
    
    from sklearn.neighbors import KNeighborsRegressor 
    
    i_count=0
    for i in np.arange(1,100,2):
    
        kn = KNeighborsRegressor(i)
        
        kn.fit(x_train[:,:,3],y_train)
        preds = kn.predict(x_test[:,:,3])
        
        corrs[n_count,i_count] = (np.corrcoef(y_test.T,preds.T)[0,1])
        i_count = i_count + 1
        
    n_count = n_count + 1
        
    print(n)
    
plt.imshow(corrs)
    
#plt.plot(y_test,preds,'.')
#plt.title('corrcoef = ' + str(np.corrcoef(y_test.T,preds.T)[0,1]))
    
    



















