# General modules
import numpy as np
import pandas as pd
# Data preprocessing tools
from keras.preprocessing.sequence import pad_sequences
# Tokenizer
import sentencepiece as sp


clean_train_filename = "clean_en_fr.txt" # both languages

subset_size = 10000 # total sentence pairs in dataset: 175,623
test_size = 1000 # sentence pairs for testing
train_size = subset_size - test_size # sentence pairs for training

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


# Load trained tokenizer for English and French
# Creating a tokenizer object for English
en_sp = sp.SentencePieceProcessor()
# Loading the English model
en_sp.Load("en.model")
# Creating a tokenizer object for French
fr_sp = sp.SentencePieceProcessor()
# Loading the French model
fr_sp.Load("fr.model")

# Load the cleaned up dataset
train_df = pd.read_csv(
    clean_train_filename,
    sep='\t')

# Function to tokenize our text (list of sentences) and
# add it to our data frame in the column 'label'
def tokenize_text(df, spm, txt_label, id_label):
    ids = []
    for line in df[txt_label].tolist():
        id_line = spm.EncodeAsIds(line)
        ids.append(id_line)
    df[id_label] = ids

# Let's run this function on the English text
tokenize_text(train_df, en_sp, 'en', 'en_ids')
# And on the French text
tokenize_text(train_df, fr_sp, 'fr', 'fr_ids')

# Check tokenized English sentence length
en_max_len = max(len(line) for line in train_df['en_ids'].tolist())
# Check tokenized French sentence length
fr_max_len = max(len(line) for line in train_df['fr_ids'].tolist())

print("fr_max_len:", fr_max_len)
print("en_max_len:", en_max_len)

# Sentence padding
# Pad English tokens
padded_en_ids = pad_sequences(
    train_df['en_ids'].tolist(),
    maxlen = en_max_len,
    padding = 'post')
# Add them to our training data frame
train_df['pad_en_ids'] = padded_en_ids.tolist()

# Pad French tokens
padded_fr_ids = pad_sequences(
    train_df['fr_ids'].tolist(),
    maxlen = fr_max_len,
    padding = 'post')
# Add them to our training data frame
train_df['pad_fr_ids'] = padded_fr_ids.tolist()

# Shuffling our dataframe around
train_df = train_df.sample(frac=1).reset_index(drop=True)

# Create our training input and target output numpy array to feed to our NMT model
# We'll take the first train_size lines for training (after random shuffle)
trainX = np.asarray(train_df['pad_fr_ids'][0:train_size].tolist())
trainY = np.asarray(train_df['pad_en_ids'][0:train_size].tolist())
# Reshape the output to match expected dimensionality
trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], 1)

# The test dataset for checking on the last test_size lines (after random shuffle)
testX = np.asarray(train_df['pad_fr_ids'][train_size:].tolist())
testY = np.asarray(train_df['pad_en_ids'][train_size:].tolist())
# Reshape the output to match expected dimensionality
testY = testY.reshape(testY.shape[0], testY.shape[1], 1)

# Saving ready-to-use numpy arrays
np.save(trainX_array_filename, trainX)
np.save(trainY_array_filename, trainY)
np.save(testX_array_filename, testX)
np.save(testY_array_filename, testY)
