# src/utils.py
import pandas as pd
import os

def save_dataframe(df: pd.DataFrame, path: str):
    """
    Save DataFrame to CSV and create parent directories if needed.
    Returns the path that was written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame. Raises FileNotFoundError if missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path, encoding="utf-8")
