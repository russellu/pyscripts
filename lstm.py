from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import mne as mne
import matplotlib.pyplot as plt 
model = Sequential()


montage = mne.channels.read_montage('standard-10-5-cap385',path='/media/sf_shared/')

raw = mne.io.read_raw_brainvision('/media/sf_shared/badger_eeg/alex/outside/alex_outside.vhdr',montage=montage,eog=[31])

raw.load_data()
raw.resample(250)

raw.filter(1,120)
x = raw.get_data()

training_length = 200; 



# Embedding layer
model.add(
    Embedding(input_dim=num_words,
              input_length = training_length,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_words, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])