# src/data_collect.py
"""
Robust tweet collector using Tweepy v2 with page-by-page partial saves.

Usage (from project root):
python -m src.data_collect --query "music" --max_results 500 --page_size 100

Behavior:
- Saves each page of tweets to PARTIAL_RAW_PATH (append mode).
- If interrupted (Ctrl+C) or rate-limited, partial results remain for analysis.
- At successful finish (or on KeyboardInterrupt), it deduplicates partial results
  and writes the final RAW_DATA_PATH CSV.
"""
import argparse
import os
import csv
import time
from typing import List, Dict
from datetime import datetime, timezone

import tweepy
import pandas as pd

from src.config import TWITTER_BEARER_TOKEN, PARTIAL_RAW_PATH, RAW_DATA_PATH, DATA_DIR
from src.utils import save_dataframe, dedupe_csv, load_dataframe

def _row_from_tweet(tweet) -> Dict:
    """
    Convert a Tweepy Tweet object to a serializable dict.
    Fields: id, created_at, author_id, text
    """
    return {
        "id": tweet.id,
        "created_at": tweet.created_at.isoformat() if hasattr(tweet, "created_at") and tweet.created_at is not None else "",
        "author_id": getattr(tweet, "author_id", ""),
        "text": getattr(tweet, "text", "")
    }

def _append_page_rows(rows: List[Dict], tmp_path: str = PARTIAL_RAW_PATH):
    """
    Append a page of rows (list of dicts) to tmp_path CSV.
    This writes header only if file doesn't exist.
    """
    os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)
    write_header = not os.path.exists(tmp_path)
    with open(tmp_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id","created_at","author_id","text"])
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def collect_tweets(query: str, max_results: int = 200, page_size: int = 100):
    """
    Collect up to max_results tweets for `query`. Saves partial pages to PARTIAL_RAW_PATH.
    Returns a DataFrame of all collected tweets (deduplicated).
    """
    if not TWITTER_BEARER_TOKEN:
        raise EnvironmentError("TWITTER_BEARER_TOKEN not set. Add it to .env or environment variables.")

    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

    tweet_fields = ["created_at", "lang", "text", "author_id"]
    query_string = f"{query} -is:retweet lang:en"

    collected = []
    collected_count = 0

    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query_string,
        tweet_fields=tweet_fields,
        max_results=page_size
    )

    try:
        for page in paginator.pages():
            # page is a Response; page.data is list of Tweet objects (or None)
            tweets_page = page.data or []
            if not tweets_page:
                # no results in this page
                continue

            # convert page tweets to serializable rows
            rows = [_row_from_tweet(t) for t in tweets_page]
            # append this page to partial CSV immediately
            _append_page_rows(rows, tmp_path=PARTIAL_RAW_PATH)

            # also accumulate in memory for function return
            collected.extend(rows)
            collected_count += len(rows)
            print(f"Collected page with {len(rows)} tweets — total so far: {collected_count}")

            # stop if we've reached requested max_results
            if collected_count >= max_results:
                print(f"Reached requested max_results={max_results}. Stopping collection loop.")
                break

        # After loop completes normally, dedupe partial CSV into final RAW_DATA_PATH
        print("Collection finished; building final deduplicated CSV.")
        # If PARTIAL_RAW_PATH exists, dedupe it to RAW_DATA_PATH
        if os.path.exists(PARTIAL_RAW_PATH):
            dedupe_csv(PARTIAL_RAW_PATH, dedupe_on=["id"], out_path=RAW_DATA_PATH)
            print(f"Wrote final deduplicated raw file to: {RAW_DATA_PATH}")
        else:
            # fallback: use collected list in memory
            if collected:
                df = pd.DataFrame(collected)
                df["collected_at"] = datetime.now(timezone.utc).isoformat()
                save_dataframe(df, RAW_DATA_PATH)
                print(f"Wrote final raw file to: {RAW_DATA_PATH}")
            else:
                print("No tweets were collected to write.")

        # return DataFrame loaded from RAW_DATA_PATH
        return load_dataframe(RAW_DATA_PATH)

    except KeyboardInterrupt:
        # user aborted with Ctrl+C — ensure partials are deduped and saved
        print("\nUser requested abort (KeyboardInterrupt). Deduplicating partial results to final CSV...")
        if os.path.exists(PARTIAL_RAW_PATH):
            dedupe_csv(PARTIAL_RAW_PATH, dedupe_on=["id"], out_path=RAW_DATA_PATH)
            print(f"Partial results saved (deduped) to: {RAW_DATA_PATH}")
            return load_dataframe(RAW_DATA_PATH)
        else:
            print("No partial file found to save. Exiting.")
            raise

    except Exception as e:
        # on any other exception (including rate-limit waits thrown by tweepy),
        # we still want to preserve partial results to disk for later analysis.
        print("Exception during collection:", repr(e))
        if os.path.exists(PARTIAL_RAW_PATH):
            try:
                dedupe_csv(PARTIAL_RAW_PATH, dedupe_on=["id"], out_path=RAW_DATA_PATH)
                print(f"Partial results saved (deduped) to: {RAW_DATA_PATH}")
            except Exception as ex2:
                print("Failed to dedupe partial file:", repr(ex2))
        raise
