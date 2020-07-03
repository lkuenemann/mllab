# 3rd party librairies
import sentencepiece as sp # Tokenizer
# Local modules
from params import * # set of all parameters


# Creating a tokenizer object for English
en_sp = sp.SentencePieceProcessor()
# Loading the English model
en_sp.Load("en.model")
# Creating a tokenizer object for French
fr_sp = sp.SentencePieceProcessor()
# Loading the French model
fr_sp.Load("fr.model")

# Testing the English tokenizer
en_test_sentence = "I like green apples."
# Encoding pieces
print(en_sp.EncodeAsPieces(en_test_sentence))
# Encoding pieces as IDs
print(en_sp.EncodeAsIds(en_test_sentence))
# Decoding encoded IDs
print(en_sp.DecodeIds(en_sp.EncodeAsIds(en_test_sentence)))

# Testing the French tokenizer
fr_test_sentence = "J'aime les pommes vertes."
# Encoding pieces
print(fr_sp.EncodeAsPieces(fr_test_sentence))
# Encoding pieces as IDs
print(fr_sp.EncodeAsIds(fr_test_sentence))
# Decoding encoded IDs
print(fr_sp.DecodeIds(fr_sp.EncodeAsIds(fr_test_sentence)))
