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
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K

montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_E_DRIVE/')
raw = mne.io.read_raw_brainvision('/media/sf_E_DRIVE/badger_eeg/alex/outside/alex_outside.vhdr',montage=montage,eog=[31])

#raw = mne.io.read_raw_eeglab('/media/sf_E_DRIVE/badger_eeg/alex/gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

raw.load_data()
raw.resample(250)


xdat = raw.get_data()
xdat = mne.filter.filter_data(xdat, 250, 1,124)

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
xx = np.linspace(xmin,xmax,28)
yy = np.linspace(ymin,ymax,28)
xv,yv = np.meshgrid(xx,yy) 

bothv = (xv.flatten(),yv.flatten())

allinterped = np.zeros((xdat.shape[1],28,28,))
for i in np.arange(0,xdat.shape[1]):
    interped = griddata(positions,xdat[inds,i],bothv)
    interped = np.reshape(interped,xv.shape)
    allinterped[i,:,:] = interped
    
allinterped = np.reshape(allinterped,(allinterped.shape[0],28,28,1))
allinterped = np.nan_to_num(allinterped)
    
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='sigmoid', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='sigmoid')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adam', loss='mean_absolute_error')
    
autoencoder.fit(allinterped[10000:50000,:,:], allinterped[10000:50000,:,:],
            epochs=100,
            batch_size=128,
            shuffle=True,
            validation_data=(allinterped[0:10000,:,:], allinterped[0:10000,:,:]))
    
    
decoded_imgs = autoencoder.predict(allinterped[0:10000,:,:])

n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    
    
    
    
    
    
    
    
    
    