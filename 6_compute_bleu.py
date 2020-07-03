# Standard librairies
import numpy as np
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
# 3rd party librairies
import sentencepiece as sp # Tokenizer
# Local modules
from params import * # set of all parameters

    
def unpad(sentences):
	unpadded_sentences = []
	for sentence in sentences:
		unpadded_sentence = []
		for token in sentence:
			if token != 0:
				unpadded_sentence.append(token)
			else:
				break
		unpadded_sentences.append(unpadded_sentence)
	return unpadded_sentences


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

# Convert to lists
listed_expectations = []
listed_predictions = []
for index in range(test_size):
	listed_expectations.append(testY[index, :, 0].tolist())
	listed_predictions.append(predictions[index, :].tolist())

# Unpad
unpadded_expectations = unpad(listed_expectations)
unpadded_predictions = unpad(listed_predictions)

# Decode
decoded_expectations = []
for sentence in unpadded_expectations:
    decoded_expectations.append(en_sp.DecodeIds(sentence))
decoded_predictions = []
for sentence in unpadded_predictions:
    decoded_predictions.append(en_sp.DecodeIds(sentence))

print(decoded_expectations)    
print(decoded_predictions)

# Retokenize to words for computing BLEU score
retok_exp = []
for sentence in decoded_expectations:
	retok_exp.append([word_tokenize(sentence)]) # using extra [] to get correct format for nltk corpus_bleu
retok_pred = []
for sentence in decoded_predictions:
	retok_pred.append(word_tokenize(sentence))

# Calculate BLEU score
print('BLEU-1: %f' % corpus_bleu(retok_exp, retok_pred, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(retok_exp, retok_pred, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(retok_exp, retok_pred, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % corpus_bleu(retok_exp, retok_pred, weights=(0.25, 0.25, 0.25, 0.25)))
