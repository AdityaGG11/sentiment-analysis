# src/config.py
import os
from dotenv import load_dotenv

# Load variables from .env (if present)
load_dotenv()

# Your Twitter token (read from .env or environment variable)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Paths (project-root-relative)
ROOT = os.path.dirname(os.path.dirname(__file__))   # folder above src/
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

# Specific file paths used by scripts
RAW_DATA_PATH = os.path.join(DATA_DIR, "tweets_raw.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "tweets_clean.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_tfidf.joblib")
