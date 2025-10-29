# scripts/save_predictions_and_chart.py
import pandas as pd, joblib, os
import matplotlib.pyplot as plt

model = joblib.load("models/logistic_merged_aug.joblib" if os.path.exists("models/logistic_merged_aug.joblib") else "models/logistic_merged.joblib")
vec = joblib.load("models/tfidf_merged_aug.joblib" if os.path.exists("models/tfidf_merged_aug.joblib") else "models/tfidf_merged.joblib")

df = pd.read_csv("data/merged_for_train.csv")
df['pred'] = model.predict(vec.transform(df['clean_text']))
out_csv = "data/analysis_merged_predictions.csv"
df.to_csv(out_csv, index=False)
counts = df['pred'].value_counts()
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(5,5))
plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Predicted sentiment distribution")
plt.savefig("outputs/merged_pred_pie.png", bbox_inches='tight')
print("Saved", out_csv, "and outputs/merged_pred_pie.png")
