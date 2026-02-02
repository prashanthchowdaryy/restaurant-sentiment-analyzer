# Importing Libraries

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load Dataset
dataset = pd.read_csv(r"C:\Users\Prashanth\Desktop\Naresh_it\AI\data\Restaurant_Reviews.tsv",
                      delimiter='\t', quoting=3)

# Text Preprocessing
corpus = []
ps = PorterStemmer()

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # keep spaces
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)  # join with space
    corpus.append(review)


# Feature Extraction (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Import ML Algorithms

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Model Dictionary

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=0),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


# Train, Predict, Evaluate

results = {}

for name, model in models.items():
    print("\n==============================")
    print(f"MODEL: {name}")
    print("==============================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Best Model

best_model = max(results, key=results.get)
print("\n🏆 BEST MODEL:", best_model)
print("Best Accuracy:", results[best_model])


# Plot Accuracy Comparison

plt.figure(figsize=(10,5))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# creating pickle file
import pickle

# Save TF-IDF Vectorizer
pickle.dump(cv, open("tfidf_vectorizer.pkl", "wb"))

# Save Best Model (SVM)
best_model = models["SVM"]  # since SVM was best
pickle.dump(best_model, open("sentiment_model.pkl", "wb"))

print("Model and Vectorizer saved successfully!")
