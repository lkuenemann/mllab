from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import Dense



# Source text database
data_filename = "./fra.txt"
# Text preprocessing
subset_size = 10000 # total sentence pairs in dataset: 175,623
test_size = 1000 # sentence pairs for testing
train_size = subset_size - test_size # sentence pairs for training
clean_en_filename = "clean_en.txt" # file to save cleaned English text data
clean_fr_filename = "clean_fr.txt" # French
clean_train_filename = "clean_en_fr.txt" # both languages
# Let's define vocabulary size for out tokenizer
vocab_size = 2000
# We'll use the same size for both languages to simplify
en_vocab_size = vocab_size
fr_vocab_size = vocab_size
# Files to save our preprocessed data numpy arrays
trainX_array_filename = "trainX.npy"
trainY_array_filename = "trainY.npy"
testX_array_filename = "testX.npy"
testY_array_filename = "testY.npy"
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
attention_cells = 64
# Training parameters
nb_epochs = 30
batch_size = 64
# Switch between different model architectures
model_type = '1-a-1'
# File name to save our trained model weights
trained_model_filename = "nmt_model_" + model_type + ".h5"



# Creating a Keras Sequential object for our NMT model
model = Sequential()

if model_type == '1-1':
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

elif model_type == '1-a-1':
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
	# Adding a softmax
	model.add((Dense(
	    attention_cells,
	    activation = 'softmax')))
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
