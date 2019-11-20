import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(x_train,y_train),(x_test,y_test) = mnist.load_data()
hot1_ytrain = to_categorical(y_train)

x_train=np.expand_dims(x_train,axis=3)

model = Sequential()
model.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='sigmoid', \
                 input_shape=[28,28,1]))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
model.add(Conv2D(64,kernel_size=(5,5),activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1000,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,hot1_ytrain,epochs=3,batch_size=20)

