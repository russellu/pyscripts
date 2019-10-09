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

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_shared/')
raw = mne.io.read_raw_eeglab('/media/sf_shared/denica/alex/denica_denbcg_gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

raw.load_data()
raw.resample(250)
raw.filter(1,120)


x = raw.get_data()

wsize = 100
step = 100
ffts1 = np.zeros((wsize,np.int(x.shape[1]/step)))
ffts2 = np.zeros((wsize,np.int(x.shape[1]/step)))
for i in np.arange(0,np.int(x.shape[1]/step)-2):
    ffts1[:,i] = x[45,i*step:i*step+wsize]
    ffts2[:,i] = np.abs(np.fft.fft(ffts1[:,i]))
        


xtrain = ffts1.transpose()
ytrain = ffts2.transpose()

RESHAPED = ffts1.shape[1]
OPTIMIZER = Adam(lr=0.00005)

model = Sequential()
model.add(Dense(xtrain.shape[1], input_shape=(RESHAPED,)))#kernel_regularizer=regularizers.l1(0.00001)
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(ytrain.shape[0]))
model.add(Activation('linear'))

model.compile(optimizer=OPTIMIZER, loss='mean_absolute_error')
historY = model.fit(xtrain.transpose(), ytrain.transpose(), batch_size=300, epochs=2000, validation_split=0.2)

predicted = model.predict(xtrain.transpose())
pred2 = model.predict(xtrain.transpose())

chan = 7; 
plt.plot(predicted[6000:8000,chan]); plt.plot(ytrain[chan,6000:8000]); plt.plot(ytrain[chan,6000:8000] - predicted[6000:8000,chan])

raw2 = mne.io.read_raw_eeglab('/media/sf_E_DRIVE/badger_eeg/russell/gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

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













