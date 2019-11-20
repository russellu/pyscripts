import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers

x_data = np.sin(np.arange(-1000,1000,0.1))
x_data = x_data + (np.random.rand(20000)-0.5)*0.2
x_data = np.reshape(x_data,[2000,10,1])
y_data = x_data[1:,:,0]
x_data = np.expand_dims(x_data[:-1,:,0],axis=2)

n_train = int(0.7*x_data.shape[0])
x_train = x_data[0:n_train,:,:]
y_train = y_data[0:n_train,:]
x_test = x_data[n_train:,:,:]
y_test = y_data[n_train:,:]

model = Sequential()
model.add(LSTM(20,input_shape=[10,1]))
model.add(Dense(10,activation='linear'))
model.compile(loss='mean_absolute_error',optimizer=optimizers.adam(lr=.01))
model.fit(x_train,y_train,epochs=10,batch_size=10)

preds = model.predict(x_test)
plt.plot(preds[15,:]); plt.plot(y_test[15,:])
