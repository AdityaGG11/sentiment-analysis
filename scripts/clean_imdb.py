import pandas as pd
import os
import re

def clean_text(t):
    t = re.sub(r"http\S+", "", str(t))
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"[^A-Za-z\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/imdb_sample.csv")
    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv("data/imdb_clean.csv", index=False)
    print(f"✅ Cleaned data saved to data/imdb_clean.csv with {len(df)} rows.")
