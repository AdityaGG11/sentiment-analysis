# src/config.py
import os
from dotenv import load_dotenv

# Load .env from project root if present
load_dotenv()

# Twitter API bearer token
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Paths (project-root relative)
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

# Specific file paths used by scripts
RAW_DATA_PATH = os.path.join(DATA_DIR, "tweets_raw.csv")          # final combined raw CSV
PARTIAL_RAW_PATH = os.path.join(DATA_DIR, "tweets_partial.csv")  # partial pages saved during collection
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "tweets_clean.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_tfidf.joblib")
