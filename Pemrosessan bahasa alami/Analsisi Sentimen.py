import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Load dataset
data = pd.read_csv('/content/data_cleaned_501_1000.csv')

# Tampilkan beberapa baris awal
data[['rating', 'cleaned_ulasan']].head()
# Buat kolom label sentimen
def rating_to_sentiment(rating):
    if rating >= 4:
        return 'positif'
    elif rating == 3:
        return 'netral'
    else:
        return 'negatif'

data['sentimen'] = data['rating'].apply(rating_to_sentiment)

# Cek distribusi label
print(data['sentimen'].value_counts())
# Input dan label
X = data['cleaned_ulasan']
y = data['sentimen']

# Split data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
# Latih model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediksi
y_pred = model.predict(X_test_tfidf)
# Evaluasi performa
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualisasi confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
import joblib

# Simpan model dan vectorizer
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

