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
raw = mne.io.read_raw_eeglab('/media/sf_shared/badger_eeg/alex/gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

raw.load_data()
raw.resample(250)

raw.filter(1,120)
x = raw.get_data()

inds = np.concatenate([np.arange(0,31),np.arange(32,64)])
newinds = np.zeros(inds.shape)
positions = np.zeros([63,2])
for i in np.arange(0,63):
    newinds[i] = np.int(montage.ch_names.index(raw.ch_names[inds[i]]))
    positions[i,:] = montage.get_pos2d()[newinds[i].astype(int),:]

x = x[0:64,:]

# try a single channel first
epoch_sz = 250
epochx = np.zeros([64,epoch_sz,x.shape[1] - epoch_sz])
epochy = np.zeros([64,x.shape[1] - epoch_sz])

for i in np.arange(0,x.shape[1] - epoch_sz-1):
    epochx[:,:,i] = x[:,i:i+epoch_sz]
    epochy[:,i] = x[:,i+epoch_sz+1]


epochx = np.reshape(epochx,[64*epoch_sz,epochx.shape[2]])

RESHAPED = epochx.shape[0]
OPTIMIZER = Adam(lr=0.00005)

model = Sequential()
model.add(Dense(32, input_shape=(RESHAPED,)))#kernel_regularizer=regularizers.l1(0.00001)
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(16, input_shape=(RESHAPED,)))#kernel_regularizer=regularizers.l1(0.00001)
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(8, input_shape=(RESHAPED,)))#kernel_regularizer=regularizers.l1(0.00001)
model.add(Activation('linear'))
model.add(Dropout(0.2))
model.add(Dense(epochy.shape[0]))
model.add(Activation('linear'))

model.compile(optimizer=OPTIMIZER, loss='mean_absolute_error')
historY = model.fit(epochx.transpose(), epochy.transpose(), 
                    batch_size=300, epochs=50, validation_split=0.2)

pred = model.predict(epochx.transpose()).transpose()

subbed = epochy - pred; 
newx = np.zeros(x.shape)
newx[:,0:subbed.shape[1]] = subbed

ch_types = []
ch_names = []
for i in np.arange(0,63):
    ch_types.append('eeg')
    ch_names.append(raw.ch_names[inds[i]])
    

info = mne.create_info(ch_names=ch_names,ch_types=ch_types,sfreq=250)
newraw = mne.io.RawArray(newx[inds,:],info)
newraw.set_montage(montage)
newraw.filter(1,120)
ica2 = ICA(n_components=40, method='fastica', random_state=23)
ica2.fit(newraw)
ica2.plot_components(picks=np.arange(0,40))










