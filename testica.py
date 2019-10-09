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
raw = mne.io.read_raw_eeglab('/media/sf_shared/badger_eeg/alex/denbcg_gradeeg_retino_gamma_01.set',montage=montage,eog=[31])

raw.load_data()
raw.resample(250)
raw.filter(1,120)
ica2 = ICA(n_components=40, method='fastica', random_state=23)
ica2.fit(raw)
ica2.plot_components(picks=np.arange(0,40))

events  = mne.find_events(raw, stim_channel='STI 014',shortest_event=1)

epochs = mne.Epochs(raw, events, event_id=1, tmin=-1, tmax=6)

#epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
#                    baseline=baseline, reject=dict(grad=4000e-13))




