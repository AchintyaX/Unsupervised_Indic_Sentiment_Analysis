import textblob
from textblob import TextBlob
import numpy as np
# retrieving the sentiment vector 
def sentiment_coeff(article):
	sentiment_vector = []
	for word in article:
		token = TextBlob(word).polarity
		sentiment_vector.append(token)

	return sentiment_vector



# retrieving the term frequency vector 
def term_frequency(article):
    term_freq = {}
    for token in article:
        term_freq[token] = article.count(token)
    term_vector = []
    for token in article:
        term_val = term_freq[token]/len(article)
        term_vector.append(term_val)
    return term_vector

def predict_sentiment(term_vector, senti_vector):
	term_vector = np.array(term_vector)
	senti_vector = np.array(senti_vector)

	polarity = np.dot(term_vector, senti_vector)

	if polarity > 0:
		return 1
	else:
		return 0