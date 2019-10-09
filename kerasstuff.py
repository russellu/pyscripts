from __future__ import print_function
from keras.models import Sequential
import numpy as np
from keras.datasets import mnist
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import regularizers
np.random.seed(1671)
NB_EPOCH = 50
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.001)
N_HIDDEN=128
VALIDATION_SPLIT=0.2

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED).astype('float32') / 255
X_test = X_test.reshape(10000, RESHAPED).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,),kernel_regularizer=regularizers.l1(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])
historY = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("score:",score[0])
print("accuracy", score[1])

#predictions = model.predict(X) 
