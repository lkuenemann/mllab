# Standard librairies
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Dense
# Local modules
from params import * # set of all parameters


# Load our ready-to-use numpy arrays for training and testing
trainX = np.load(trainX_array_filename)
trainY = np.load(trainY_array_filename)
testX = np.load(testX_array_filename)
testY = np.load(testY_array_filename)

# Creating a Keras Sequential object for our NMT model
model = Sequential()

# Embedding layer to map our one-hot encoding to a small word space
model.add(Embedding(
    fr_vocab_size,
    nb_cells,
    input_length = fr_max_len,
    mask_zero = True))
# Adding an LSTM layer to act as the encoder
model.add(LSTM(
    units = nb_cells,
    return_sequences = False))
# Since we are not returning a sequence but just a vector, we need
# to repeat this vector multiple times to input it to our decoder LSTM
model.add(RepeatVector(en_max_len))
# Adding an LSTM layer to act as the decoder
model.add(LSTM(
    units = nb_cells,
    return_sequences = True))
# Adding a softmax
model.add((Dense(
    en_vocab_size,
    activation = 'softmax')))

# Compiling the model
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy')

print(model.summary())

# Training the model
checkpoint = ModelCheckpoint(
    trained_model_filename,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    mode = 'min')
model.fit(
    trainX,
    trainY,
    epochs = nb_epochs,
    batch_size = batch_size,
    callbacks = [checkpoint],
    validation_data = (testX, testY))
