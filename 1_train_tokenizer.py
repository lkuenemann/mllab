# Tokenizer
import sentencepiece as sp

clean_en_filename = "clean_en.txt" # file to save cleaned English text data
clean_fr_filename = "clean_fr.txt" # French
# Let's define vocabulary size for out tokenizer
vocab_size = 2000
# We'll use the same size for both languages to simplify
en_vocab_size = vocab_size
fr_vocab_size = vocab_size

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
