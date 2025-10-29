# scripts/train_tfidf_merged.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Load merged dataset
df = pd.read_csv("data/merged_for_train.csv")

# Check columns
print("Columns:", df.columns.tolist())

# Choose text + label columns (adjust based on your merged data)
text_col = "clean_text" if "clean_text" in df.columns else "text"
label_col = "auto_sentiment" if "auto_sentiment" in df.columns else "sentiment"

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logistic_merged.joblib")
joblib.dump(vectorizer, "models/tfidf_merged.joblib")

print("\n✅ Model and vectorizer saved to 'models/'")
