# src/utils.py
"""
Utility helpers for saving/loading dataframes.
Provides save_dataframe with optional append + dedupe behavior.
"""
import os
import pandas as pd
from typing import Optional, List

def save_dataframe(df: pd.DataFrame, path: str, append: bool = False, dedupe_on: Optional[List[str]] = None) -> str:
    """
    Save DataFrame to CSV.

    - path: destination CSV path.
    - append: if True, append to existing CSV (will create parent dir if needed).
    - dedupe_on: if provided and append=True, will drop duplicates based on the given column(s)
                 after concatenation (keeps first occurrence).

    Returns the path written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if append and os.path.exists(path):
        # Read existing, concat, optionally dedupe, then write fresh (to keep header order consistent)
        existing = pd.read_csv(path, encoding="utf-8")
        combined = pd.concat([existing, df], ignore_index=True)
        if dedupe_on:
            combined = combined.drop_duplicates(subset=dedupe_on)
        combined.to_csv(path, index=False, encoding="utf-8")
    elif append and not os.path.exists(path):
        # simply write with header
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        # overwrite mode
        df.to_csv(path, index=False, encoding="utf-8")
    return path

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load CSV into a pandas DataFrame. Raises FileNotFoundError if path not present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path, encoding="utf-8")

def dedupe_csv(path: str, dedupe_on: List[str], out_path: Optional[str] = None) -> str:
    """
    Read CSV from path, drop duplicates based on dedupe_on, save back to either out_path or path.
    Returns the final path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop_duplicates(subset=dedupe_on)
    target = out_path if out_path else path
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8")
    return target
