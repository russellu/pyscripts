import numpy as np
import matplotlib.pyplot as plt 
from keras.layers import Input, Dense 
from keras.models import Sequential
from sklearn.datasets import load_breast_cancer
from sklearn import svm

cancer = load_breast_cancer()

clf = svm.SVC(gamma='scale')
clf.fit(cancer.data,cancer.target)
preds = clf.predict(cancer.data)
svm_acc = 1-np.sum(np.abs(preds-cancer.target))/cancer.target.shape[0]

model = Sequential()

model.add(Dense(5,activation='relu',input_dim=30))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(cancer.data,cancer.target,epochs=150,batch_size=10)

_,nn_acc=model.evaluate(cancer.data,cancer.target)



