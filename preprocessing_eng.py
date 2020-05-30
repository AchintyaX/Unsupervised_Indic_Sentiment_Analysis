import nltk 
import re 
import string 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# removing named entities 
def remove_named_entities(text):
    doc = nlp(text)
    words = text.split()
    named_entities = [X.text for X in doc.ents]
    words = [i for i in words if not i in named_entities ]
    result = ' '.join(words)
    return result 


# removing punctuations and numbers 
def preprocess(text):
    text = text.lower()
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)


    return text

# tokenize and remove left over named entities as well as lemmatize
def tokenize_named_entities_removal(text):
    doc = nlp(text)
    tokens = [ token.lemma_ for token in doc]
    tokens = [i for i in tokens if  i != '-PRON-' ]
    named_entities = [X.text for X in doc.ents]
    words = [i for i in tokens if not i in named_entities]


# removing stop words 
def remove_stopwords(article):
    stopwords = nltk.corpus.stopwords.words('english')
    result = [i for i in article if not i in stopwords]
    return result 