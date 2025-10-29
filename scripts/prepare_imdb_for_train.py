# scripts/prepare_imdb_for_train.py
import pandas as pd
import os

IN = "data/imdb_clean.csv"
OUT = "data/imdb_for_train.csv"

df = pd.read_csv(IN)

# If dataset has a numeric 'label' column (0/1), map to text labels
if 'label' in df.columns and 'sentiment' not in df.columns:
    df['sentiment'] = df['label'].map({0: 'negative', 1: 'positive'})

# If sentiment exists, copy it to auto_sentiment (train_model expects this)
if 'sentiment' in df.columns:
    df['auto_sentiment'] = df['sentiment'].astype(str)
elif 'label' in df.columns:
    # fallback: numeric mapping
    df['auto_sentiment'] = df['label'].map({0: 'negative', 1: 'positive'}).astype(str)
else:
    raise SystemExit("Input CSV missing 'sentiment' or 'label' column. Open data/imdb_clean.csv to check columns.")

# Ensure clean_text exists (train_model expects it)
if 'clean_text' not in df.columns and 'text' in df.columns:
    # minimal cleaning: copy text -> clean_text
    df['clean_text'] = df['text'].astype(str)

# Save the file used for training
os.makedirs('data', exist_ok=True)
df.to_csv(OUT, index=False, encoding='utf-8')
print(f"Wrote {OUT} with columns: {list(df.columns)}; rows={len(df)}")
