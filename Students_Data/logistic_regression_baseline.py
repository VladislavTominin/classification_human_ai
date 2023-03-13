import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.metrics import classification_report
from human_ai.code.evaluate import save_predictions

# Read The data
training_set = pd.read_json('./train_set.json')
test_set = pd.read_json('./test_set.json')

# Use logistic regression to predict the class
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(training_set['text'])
clf = LogisticRegression()
clf.fit(X, training_set['label'])
X_test = vectorizer.transform(test_set['text'])
predictions = clf.predict(X_test)

save_predictions(predictions)