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

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_E_DRIVE/')
raw = mne.io.read_raw_brainvision('/media/sf_E_DRIVE/badger_eeg/alex/outside/alex_outside.vhdr',montage=montage,eog=[31])

#raw = mne.io.read_raw_eeglab('/media/sf_E_DRIVE/badger_eeg/alex/gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

raw.load_data()
raw.resample(250)


x = raw.get_data()
x = mne.filter.filter_data(x, 250, 1,124)

inds = np.concatenate([np.arange(0,31),np.arange(32,64)])
newinds = np.zeros(inds.shape)
positions = np.zeros([63,2])
for i in np.arange(0,63):
    newinds[i] = np.int(montage.ch_names.index(raw.ch_names[inds[i]]))
    positions[i,:] = montage.get_pos2d()[newinds[i].astype(int),:]

xmax = np.max(positions[:,0])
xmin = np.min(positions[:,0])
ymax = np.max(positions[:,1])
ymin = np.min(positions[:,1])
xx = np.linspace(xmin,xmax,32)
yy = np.linspace(ymin,ymax,32)
xv,yv = np.meshgrid(xx,yy) 

bothv = (xv.flatten(),yv.flatten())

allinterped = np.zeros((32,32,x.shape[1]))
for i in np.arange(0,x.shape[1]):
    interped = griddata(positions,x[inds,i],bothv)
    interped = np.reshape(interped,xv.shape)
    allinterped[:,:,i] = interped

xtrain = x[inds,5000:]
ytrain = x[inds,5000:]

RESHAPED = 63
OPTIMIZER = Adam(lr=0.00005)

model = Sequential()
model.add(Dense(60, input_shape=(RESHAPED,)))#kernel_regularizer=regularizers.l1(0.00001)
model.add(Activation('linear'))
model.add(Dropout(0.8))
model.add(Dense(63))
model.add(Activation('linear'))

model.compile(optimizer=OPTIMIZER, loss='mean_absolute_error')
historY = model.fit(xtrain.transpose(), ytrain.transpose(), batch_size=250, epochs=200, validation_split=0.2)

prediction = model.predict(xtrain.transpose())

weights = model.layers[3].get_weights()[0]

for i in np.arange(1,61):
    plt.subplot(6,10,i); 
    mne.viz.plot_topomap(weights[i-1,:],positions)
    
    
layer_name = 'activation_17'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
out = intermediate_layer_model.predict(xtrain.transpose())

f,pxx = signal.welch(out.transpose(),250,nperseg=500)
f,pxxraw = signal.welch(xtrain,250,nperseg=500)
    
    
"""
fig = plt.figure()
i=0
im = plt.imshow(allinterped[:,:,0], animated=True)
def updatefig(*args):
    global i
    if (i<99):
        i += 1
    else:
        i=0
    im.set_array(allinterped[:,:,i])
    return im,
ani = animation.FuncAnimation(fig, updatefig)
plt.show()
"""




















