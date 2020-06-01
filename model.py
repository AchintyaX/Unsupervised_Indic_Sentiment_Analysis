import textblob
from textblob import TextBlob


def sentiment_coeff(article):
	sentiment_vector = []
	for word in article:
		token = TextBlob(word).polarity
		sentiment_vector.append(token)

	return sentiment_vector