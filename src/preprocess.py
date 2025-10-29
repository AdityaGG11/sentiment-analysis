# src/preprocess.py
"""
Preprocess tweets: cleaning and weak auto-labeling (TextBlob) for quick experiments.
Saves cleaned CSV to data/tweets_clean.csv via src.utils.save_dataframe.
"""
import re
import pandas as pd
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH
from src.utils import save_dataframe, load_dataframe

# Ensure stopwords are available (silent if already downloaded)
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Clean a tweet:
      - remove URLs
      - remove mentions, keep hashtag words by removing '#'
      - remove non-alphanumeric characters (keeps spaces and apostrophes)
      - lowercase, normalize whitespace
      - remove English stopwords
    Returns cleaned string.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.IGNORECASE)
    # Remove mentions
    text = re.sub(r"@\w+", " ", text)
    # Remove '#' but keep the tag text
    text = re.sub(r"#", "", text)
    # Remove unusual punctuation (keep apostrophes and alphanumerics and spaces)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove stopwords
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def auto_label_text(text: str) -> str:
    """
    Weak automatic labeling using TextBlob polarity:
      polarity > 0.1 -> 'positive'
      polarity < -0.1 -> 'negative'
      otherwise -> 'neutral'
    This is only for bootstrapping; prefer human labels later.
    """
    try:
        polarity = TextBlob(text).sentiment.polarity
    except Exception:
        polarity = 0.0
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def preprocess_and_save(raw_path: str = RAW_DATA_PATH, clean_path: str = CLEAN_DATA_PATH):
    """
    Main entrypoint used by scripts:
      - loads raw CSV at raw_path (expects a 'text' column)
      - applies clean_text and creates 'clean_text' column
      - creates 'auto_sentiment' weak labels
      - saves cleaned CSV to clean_path using save_dataframe
    """
    # Load raw data (will raise FileNotFoundError if missing)
    df = load_dataframe(raw_path)

    # Create cleaned text column
    df["clean_text"] = df["text"].fillna("").apply(clean_text)

    # Auto-label for quick experiments
    df["auto_sentiment"] = df["clean_text"].apply(auto_label_text)

    # Save cleaned DataFrame
    save_dataframe(df, clean_path)
    print(f"Saved cleaned data to {clean_path}")

# When script run directly, call the function
if __name__ == "__main__":
    preprocess_and_save()
