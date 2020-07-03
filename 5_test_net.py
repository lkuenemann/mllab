# General modules
import numpy as np
# General Keras functionalities
from keras.models import load_model
# Tokenizer
import sentencepiece as sp


# Files to save our preprocessed data numpy arrays
testX_array_filename = "testX.npy"
testY_array_filename = "testY.npy"
# File name to save our trained model weights
trained_model_filename = "fr_en_nmt_model_test.h5"


# Let's load the trained model
model = load_model(trained_model_filename)

# Load our ready-to-use numpy arrays for testing
testX = np.load(testX_array_filename)
testY = np.load(testY_array_filename)

# Load trained tokenizer for English and French
# Creating a tokenizer object for English
en_sp = sp.SentencePieceProcessor()
# Loading the English model
en_sp.Load("en.model")
# Creating a tokenizer object for French
fr_sp = sp.SentencePieceProcessor()
# Loading the French model
fr_sp.Load("fr.model")

# Predict
predictions = model.predict_classes(testX)

# Check the translation on a few sentences
decoded_predictions = []
for index in range(10):
    print("Original:")
    print(fr_sp.DecodeIds(testX[index, :].tolist()))
    print("Expected:")
    print(en_sp.DecodeIds(testY[index, :, 0].tolist()))
    print("Predicted:")
    print(en_sp.DecodeIds(predictions[index, :].tolist()))
    print("")
