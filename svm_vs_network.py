
"""
SVM vs neural network
effects of scaling data
test set and training set
overfitting 

"""

import numpy as np
import matplotlib.pyplot as plt 
from keras.layers import Input, Dense 
from keras.models import Sequential
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn import svm

cancer = load_breast_cancer()
input_data = cancer.data
input_data = scale(input_data)


# sklearn support vector machine 
clf = svm.SVC(gamma='scale')
clf.fit(input_data,cancer.target)
preds = clf.predict(input_data)
svm_acc = 1-np.sum(np.abs(preds-cancer.target))/cancer.target.shape[0]

# neural network in keras
model = Sequential()
model.add(Dense(1,activation='sigmoid',input_dim=30))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(input_data,cancer.target,epochs=150,batch_size=10)
_,nn_acc=model.evaluate(input_data,cancer.target)








