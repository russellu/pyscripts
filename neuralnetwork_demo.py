import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Sequential 

n_samps = 1000
attribute_1 = np.round(np.random.rand(n_samps,1))
attribute_2 = np.round(np.random.rand(n_samps,1))

input_data = np.zeros([n_samps,2])
input_data[:,0] = attribute_1[:,0]
input_data[:,1] = attribute_2[:,0]

output = np.sum(input_data,axis=1) == 1
output = output.astype(float)

model = Sequential()
model.add(Dense(2,input_dim=2,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(input_data,output,epochs=100,batch_size=10)


from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(input_data,output)
preds = clf.predict(input_data)
svm_acc = 1 - np.sum(np.abs(preds - output))/output.shape[0]




