# scripts/fetch_imdb.py
# Lightweight: downloads a small IMDB sample and saves as CSV for quick training.
from datasets import load_dataset
import pandas as pd
import os

print("Downloading IMDB dataset (only a small sample will be saved)...")
ds = load_dataset("imdb")  # will download the dataset the first time

# Take a small sample to keep this laptop-friendly
n_samples = 2000
train_df = pd.DataFrame(ds["train"]).sample(n=min(n_samples, len(ds["train"])), random_state=42)

# Map numeric labels to text labels
train_df["sentiment"] = train_df["label"].map({0: "negative", 1: "positive"})

# Keep only the columns we need and save
out = train_df[["text", "sentiment"]].reset_index(drop=True)
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "imdb_sample.csv")
out.to_csv(out_path, index=False, encoding="utf-8")
print(f"Saved {len(out)} rows to {out_path}")
