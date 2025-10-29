# scripts/diagnose_model.py
import pandas as pd, joblib, os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns, matplotlib.pyplot as plt

df = pd.read_csv("data/merged_for_train.csv")
vec = joblib.load("models/tfidf_merged.joblib")
model = joblib.load("models/logistic_merged.joblib")

X = vec.transform(df['clean_text'])
y_true = df['auto_sentiment'].astype(str).tolist()
y_pred = model.predict(X)

print("CLASS COUNTS\n", pd.Series(y_true).value_counts())
print("\nCLASSIFICATION REPORT\n", classification_report(y_true, y_pred, zero_division=0))

# confusion matrix plot
labels = sorted(list(set(y_true)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
print("\nSaved confusion matrix to outputs/confusion_matrix.png")
