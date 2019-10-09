import mne as mne
import matplotlib.pyplot as plt 
from keras.models import Sequential, Model
import numpy as np
from keras.datasets import mnist
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import regularizers
from scipy import signal
from scipy.interpolate import griddata
import matplotlib.animation as animation
from mne.preprocessing import ICA

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_E_DRIVE/')
raw = mne.io.read_raw_eeglab('/media/sf_E_DRIVE/badger_eeg/alex/gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

raw.load_data()
raw.resample(250)
raw.filter(1,120)

x = raw.get_data()

ica = ICA(n_components=35, method='fastica', random_state=23)
ica.fit(raw)
src = ica.get_sources(raw)
srcdata = src.get_data()

inds = np.concatenate([np.arange(0,31),np.arange(32,64)])
newinds = np.zeros(inds.shape)
positions = np.zeros([63,2])
for i in np.arange(0,63):
    newinds[i] = np.int(montage.ch_names.index(raw.ch_names[inds[i]]))
    positions[i,:] = montage.get_pos2d()[newinds[i].astype(int),:]

lag = 400
lag_pulse = np.zeros((lag,x.shape[1]))
for i in np.arange(lag,x.shape[1]-lag):
    lag_pulse[:,i] = x[31,i-lag:i] 
    
xtrain = srcdata[0:5,5000:]/100000 
ytrain = x[inds,5000:]

RESHAPED = 5
OPTIMIZER = Adam(lr=0.00005)

model = Sequential()
model.add(Dense(60, input_shape=(5,)))#kernel_regularizer=regularizers.l1(0.00001)
model.add(Activation('linear'))
model.add(Dropout(0.3))
model.add(Dense(63))
model.add(Activation('linear'))

model.compile(optimizer=OPTIMIZER, loss='mean_absolute_error')
historY = model.fit(xtrain.transpose(), ytrain.transpose(), batch_size=300, epochs=1000, validation_split=0.2)

prediction = model.predict(xtrain.transpose())
predicted = model.predict(xtrain.transpose())
pred2 = model.predict(srcdata[0:5,:].transpose()/100000)



#f,pxx = signal.welch(out.transpose(),250,nperseg=500)
#f,pxxraw = signal.welch(xtrain,250,nperseg=500)
    
chan = 0; 
plt.plot(predicted[6000:8000,chan]); plt.plot(ytrain[chan,6000:8000]); plt.plot(ytrain[chan,6000:8000] - predicted[6000:8000,chan])

raw2 = mne.io.read_raw_eeglab('/media/sf_E_DRIVE/badger_eeg/alex/gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

subbed = raw2.get_data()[inds,:] - pred2.transpose()
raw2.get_data()[inds,:] = subbed

ch_types = []
ch_names = []
for i in np.arange(0,63):
    ch_types.append('eeg')
    ch_names.append(raw.ch_names[inds[i]])
    

info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=250)
newraw = mne.io.RawArray(subbed,info)
newraw.set_montage(montage)
newraw.filter(1,120)
ica2 = ICA(n_components=60, method='fastica', random_state=23)
ica2.fit(newraw)
ica2.plot_components(picks=np.arange(0,60))


"""
weights = model.layers[3].get_weights()[0]

for i in np.arange(1,61):
    plt.subplot(6,10,i); 
    mne.viz.plot_topomap(weights[i-1,:],positions)
    
    
layer_name = 'activation_3'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
out = intermediate_layer_model.predict(xtrain.transpose())
"""


















