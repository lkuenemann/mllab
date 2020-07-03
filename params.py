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

# Training parameters
nb_epochs = 30
batch_size = 64

# File name to save our trained model weights
trained_model_filename = "fr_en_nmt_model_test.h5"

