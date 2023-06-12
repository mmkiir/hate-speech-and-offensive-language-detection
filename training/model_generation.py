import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from utils import preprocess_data

data = pd.read_csv("labeled_data.csv")
text, labels, vocab_size = preprocess_data(data)
X_train, x_test, y_train, y_test = train_test_split(
    text, labels, test_size=0.2, random_state=42
)


naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

with open("models/naive_bayes.pkl", "wb") as f:
    pickle.dump(naive_bayes, f)
print("Naive Bayes model saved")

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

with open("models/decision_tree.pkl", "wb") as f:
    pickle.dump(decision_tree, f)
print("Decision Tree model saved")
