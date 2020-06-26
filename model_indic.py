from gensim.models.fasttext import FastText


# Getting the polarity of each word 
def get_polarity(word, word_list, model):
    num_words = len(word_list)
    score = 0
    for i in word_list:
        try:
            similarity = model.wv.similarity(word, i['word']) 
        except KeyError:
            similarity = 0
        if i['pos']>i['neg']:
            coeff = i['pos']
        else:
            coeff = i['neg']
        score = score + (similarity*coeff)
    score = score/num_words 

    return score 

# Getting the sentiment vectors of each sentence 

def get_senti_coeff_indic(article, pos_words, neg_words, model):
    sentiment_vector = []
    for word in article:
        token_pos = get_sentiment(word, pos_words, model)
        token_neg = get_sentiment(word, neg_words, model)
        if token_pos >= token_neg:
            token = token_pos
            if token < 0.40:
                token = 0
        else:
            token = -1*token_neg
            if token > -0.40:
                token = 0 
        sentiment_vector.append(token)
    return sentiment_vector

# updated version of the  get polarity function
def get_sentiment(word, word_list, model):
    max_similarity = 0
    for i in word_list:
        try:
            similarity = model.wv.similarity(word, i['word'])
        except KeyError:
            similarity = 0 
        if i['pos']> i['neg']:
            coeff = i['pos']
        else:
            coeff = i['neg']
        score = similarity*coeff
        if score > max_similarity:
            max_similarity = score
    return  max_similarity