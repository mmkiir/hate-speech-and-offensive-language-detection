import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils import preprocess_data

data = pd.read_csv("labeled_data.csv")
text, labels, vocab_size = preprocess_data(data)

X_train, X_test, y_train, y_test = train_test_split(
    text, labels, test_size=0.2, random_state=42
)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

num_classes = len(set(labels))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_train_cat = np.array(y_train_cat)
y_test_cat = to_categorical(y_test, num_classes=num_classes)
y_test_cat = np.array(y_test_cat)

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=150))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)

model.summary()
model.fit(
    X_train,
    y_train_cat,
    batch_size=1024,
    epochs=10,
    validation_data=(X_test, y_test_cat),
    callbacks=[es],
)

model.save("models/lstm.h5")
