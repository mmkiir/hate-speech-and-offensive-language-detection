import re
import string

import nltk
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import Bidirectional

nltk.download("stopwords")


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


def create_model_2(vocab_size, max_length, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def train_model_2(model, X_train, y_train, X_test, y_test):
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
    model.summary()
    model.fit(
        X_train,
        y_train,
        batch_size=1024,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=[es],
    )


def create_model_3(vocab_size, max_length, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def train_model_3(model, X_train, y_train, X_test, y_test):
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
    model.summary()
    model.fit(
        X_train,
        y_train,
        batch_size=1024,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=[es],
    )


# Read the data
data = pd.read_csv("labeled_data.csv")

# Preprocess the data
text, labels, vocab_size = preprocess_data(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    text, labels, test_size=0.2, random_state=42
)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

num_classes = len(set(labels))
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)
y_train_categorical = np.array(y_train_categorical)
y_test_categorical = np.array(y_test_categorical)

# Create and train model 1 (Naive Bayes)
model_1 = MultinomialNB()
model_1.fit(X_train, y_train)

# Create and train model 2 (LSTM)
model_2 = create_model_2(vocab_size, max_length=150, num_classes=num_classes)
train_model_2(model_2, X_train, y_train_categorical, X_test, y_test_categorical)

# Create and train model 3 (BiLSTM)
model_3 = create_model_3(vocab_size, max_length=150, num_classes=num_classes)
train_model_3(model_3, X_train, y_train_categorical, X_test, y_test_categorical)

# Get the accuracy of all models
print(f"Model 1 (Naive Bayes) accuracy: {model_1.score(X_test, y_test)}")
print(f"Model 2 (LSTM) accuracy: {model_2.evaluate(X_test, y_test_categorical)[1]}")
print(f"Model 3 (BiLSTM) accuracy: {model_3.evaluate(X_test, y_test_categorical)[1]}")
