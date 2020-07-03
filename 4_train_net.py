# General modules
import numpy as np
# General Keras functionalities
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
# Neural network components from Keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Dense


# Let's define vocabulary size for out tokenizer
vocab_size = 2000
# We'll use the same size for both languages to simplify
en_vocab_size = vocab_size
fr_vocab_size = vocab_size
# max_sentence_length = 20 # Pad all sentences to 40 word pieces (tokens) max
fr_max_len = 22
en_max_len = 10

# Defining all the parameters of our network
nb_cells = 256 # LSTM cells in encoder/decoder

# Training parameters
nb_epochs = 30
batch_size = 64

# Files to save our preprocessed data numpy arrays
trainX_array_filename = "trainX.npy"
trainY_array_filename = "trainY.npy"
testX_array_filename = "testX.npy"
testY_array_filename = "testY.npy"


# File name to save our trained model weights
trained_model_filename = "fr_en_nmt_model_test.h5"


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
