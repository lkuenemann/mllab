# 3rd party libraries
import sentencepiece as sp # Tokenizer
# Local modules
from params import * # set of all parameters


# Training the model for English
sp.SentencePieceTrainer.train(
    input = clean_en_filename,
    model_prefix = 'en',
    vocab_size = en_vocab_size,
)

# Training the model for French
sp.SentencePieceTrainer.train(
    input = clean_fr_filename,
    model_prefix = 'fr',
    vocab_size = fr_vocab_size,
)
