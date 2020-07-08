# Standard librairies
import numpy as np
from keras.callbacks import ModelCheckpoint
# Local modules
from params import * # set of all parameters


# Load our ready-to-use numpy arrays for training and testing
trainX = np.load(trainX_array_filename)
trainY = np.load(trainY_array_filename)
testX = np.load(testX_array_filename)
testY = np.load(testY_array_filename)



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
