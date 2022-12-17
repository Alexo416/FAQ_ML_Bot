"""
This application will load the sentiment corpus to be used by the bot to detect if utterances are negative
"""
### Load docs and labels
filenames = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
docs = []
labels = []
for filename in filenames:
    with open("sentiment/"+filename) as file:
        for line in file:
            line = line.strip()
            labels.append(int(line[-1]))
            docs.append(line[:-2].strip())

## Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(docs)

## Train
from sklearn.neural_network import MLPClassifier ##MLP Classifier is used to maximize results
clf = MLPClassifier()
clf.fit(vectors, labels)

## Pickle
from joblib import dump
dump(clf, "clf.joblib")
dump(vectorizer, "vectorizer.joblib")




