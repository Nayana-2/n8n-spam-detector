import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'text']

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Features and target
X = df['text']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model and vectorizer
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model Saved Successfully!")