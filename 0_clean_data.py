import pandas as pd

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

# Importing the dataset in a Pandas dataframe
train_df = pd.read_csv(
    data_filename, # path to our dataset file
    sep='\t', # tab delimiter between columns in the csv
    usecols=[0, 1], # import only columns 0 and 1
    nrows=subset_size, # read only the first subset_size rows
    names=["en","fr"]) # label them 'en' and 'fr'

# Saving our cleaned and reduced dataset
train_df.to_csv(
    clean_train_filename,
    sep='\t', # using tab separators
    index=False) # don't print the row index in the csv

# Saving the English part separately for SentencePiece
train_df.to_csv(
    clean_en_filename,
    columns=['en'], # print only the column 'en'
    index=False)

# And the French one
train_df.to_csv(
    clean_fr_filename,
    columns=['fr'], # print only the column 'fr'
    index=False)
