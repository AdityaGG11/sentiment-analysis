# scripts/merge_imdb_and_tweets.py
"""
Merge IMDB sample (data/imdb_clean.csv or imdb_for_train.csv) and your tweets (data/tweets_clean.csv or tweets_clean_balanced.csv)
Create a single train CSV data/merged_for_train.csv with columns: clean_text, auto_sentiment (values positive/negative/neutral)
Options: downsample largest domain to balance.
"""

import pandas as pd, os
from sklearn.utils import resample

# === CONFIG ===
IMDB_CSV = "data/imdb_for_train.csv"           # produced earlier by prepare_imdb_for_train.py
TWEETS_CSV = "data/tweets_clean.csv"          # your existing cleaned tweets file
OUT = "data/merged_for_train.csv"
TARGET_ROWS_PER_CLASS_PER_DOMAIN = 500        # adjust down to keep lightweight (set smaller if you want)
DOWNSAMPLE_IMDB = True                        # if True, downsample IMDB to avoid domination
# =============

def load_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

print("Loading datasets...")
imdb = load_if_exists(IMDB_CSV)
tweets = load_if_exists(TWEETS_CSV)

if imdb is None and tweets is None:
    raise SystemExit("No source files found. Put imdb_for_train.csv and/or tweets_clean.csv in data/")

# Normalize column names and label columns
def normalize(df, text_col_candidates):
    # choose text column
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    else:
        raise ValueError("No text column found in df: tried " + str(text_col_candidates))
    # map sentiment column
    if "auto_sentiment" in df.columns:
        label_col = "auto_sentiment"
    elif "sentiment" in df.columns:
        label_col = "sentiment"
    elif "label" in df.columns:
        # numeric -> mapping if needed
        if df['label'].dtype.kind in 'biufc':
            df['label'] = df['label'].map({0:'negative',1:'positive'}).astype(str)
        label_col = "label"
    elif "final_label" in df.columns:
        label_col = "final_label"
    else:
        raise ValueError("No label column found in df")
    # create cleaned output
    out = df[[text_col]].rename(columns={text_col: "clean_text"}).copy()
    out["auto_sentiment"] = df[label_col].astype(str)
    return out

frames = []
if imdb is not None:
    print("IMDB rows:", len(imdb))
    imdb_norm = normalize(imdb, ["clean_text","text"])
    imdb_norm["domain"] = "imdb"
    frames.append(imdb_norm)
if tweets is not None:
    print("Tweets rows:", len(tweets))
    tweets_norm = normalize(tweets, ["clean_text","text"])
    tweets_norm["domain"] = "tweets"
    frames.append(tweets_norm)

merged = pd.concat(frames, ignore_index=True)
print("Merged total rows:", len(merged))
# drop NA
merged = merged.dropna(subset=["clean_text","auto_sentiment"]).reset_index(drop=True)

# Optionally downsample IMDB if it dominates
if DOWNSAMPLE_IMDB and "imdb" in merged['domain'].unique():
    balanced_frames = []
    for dom in merged['domain'].unique():
        dom_df = merged[merged['domain'] == dom]
        # for each sentiment class in domain, resample to target rows
        for cls in dom_df['auto_sentiment'].unique():
            cls_df = dom_df[dom_df['auto_sentiment'] == cls]
            target = min(len(cls_df), TARGET_ROWS_PER_CLASS_PER_DOMAIN)
            if len(cls_df) > target:
                cls_sample = resample(cls_df, replace=False, n_samples=target, random_state=42)
            else:
                cls_sample = cls_df
            balanced_frames.append(cls_sample)
    merged = pd.concat(balanced_frames, ignore_index=True)
    print("After per-domain per-class downsampling, rows:", len(merged))

# Shuffle and save
merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
os.makedirs("data", exist_ok=True)
merged.to_csv(OUT, index=False, encoding="utf-8")
print("Wrote merged dataset to", OUT)
print(merged['auto_sentiment'].value_counts())
print("Domains:", merged['domain'].value_counts())
