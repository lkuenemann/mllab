# Standard librairies
import pandas as pd
# Local modules
from params import * # set of all parameters


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
