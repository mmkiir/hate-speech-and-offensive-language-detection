import re
import string

import nltk
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical


def clean_text(text):
    # Remove "RT @name"
    text = re.sub(r"RT @\w+", "", text)

    # Remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove whitespaces
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    text = " ".join([word for word in text.split() if word not in stopwords])

    # Stemming
    stemmer = nltk.stem.PorterStemmer()
    text = " ".join([stemmer.stem(word) for word in text.split()])

    return text


def preprocess_data(data):
    # Clean the text
    text = [clean_text(t) for t in data["tweet"].tolist()]

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    vocab_size = len(tokenizer.word_index) + 1

    # Pad the text
    max_length = 150
    text = pad_sequences(text, maxlen=max_length, padding="post")

    # Get the labels
    labels = data["class"].tolist()

    return text, labels, vocab_size
