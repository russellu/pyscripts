import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input,Dense
from keras.models import Sequential
from keras.utils import to_categorical


(x_train,y_train),(x_test,y_test) = mnist.load_data()



































"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.utils import to_categorical


(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = np.reshape(x_train,[60000,28*28])
hot1_ytrain = to_categorical(y_train)

model = Sequential()
model.add(Dense(10,activation='sigmoid',input_dim=784))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,hot1_ytrain,epochs=3,batch_size=10)

weights = model.get_weights()
reshape_weights = np.reshape(weights[0],[28,28,10])
"""