# src/data_collect.py
"""
Collect tweets using Tweepy v2 Client and save to data/tweets_raw.csv

Usage (example):
python src/data_collect.py --query "music" --max_results 100
"""
import argparse
import pandas as pd
import tweepy
from datetime import datetime
from src.config import TWITTER_BEARER_TOKEN, RAW_DATA_PATH
from src.utils import save_dataframe

def collect_tweets(query: str, max_results: int = 100):
    """
    Collect recent tweets matching `query`. Returns a DataFrame.
    This requests English tweets and filters out retweets.
    """
    # 1) Ensure token is present
    if not TWITTER_BEARER_TOKEN:
        raise EnvironmentError("TWITTER_BEARER_TOKEN not set. Add it to your .env or environment variables.")

    # 2) Create Tweepy client that will add Authorization headers for us
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

    # 3) Which tweet fields we want back
    tweet_fields = ["created_at", "lang", "text", "author_id"]

    tweets = []
    # 4) Use Tweepy Paginator to request pages of results (max_results per page default)
    #    We add "-is:retweet lang:en" to:
    #      - exclude retweets, and
    #      - restrict to English tweets (helps preprocessing)
    query_string = f"{query} -is:retweet lang:en"

    # Paginator yields Tweet objects one-by-one with .flatten(limit=...) convenience
    for tweet in tweepy.Paginator(
            client.search_recent_tweets,
            query=query_string,
            tweet_fields=tweet_fields,
            max_results=100  # 100 is the largest page size allowed in many endpoints
        ).flatten(limit=max_results):
        tweets.append({
            "id": tweet.id,
            "created_at": tweet.created_at,
            "author_id": tweet.author_id,
            "text": tweet.text
        })

    # 5) Put results in a DataFrame and add a collection timestamp
    df = pd.DataFrame(tweets)
    df["collected_at"] = datetime.utcnow().isoformat()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect recent tweets and save to CSV")
    parser.add_argument("--query", required=True, type=str, help="Search query (e.g., 'music')")
    parser.add_argument("--max_results", default=200, type=int, help="Maximum number of tweets to fetch (use small number for testing)")
    args = parser.parse_args()

    # Run collection
    df = collect_tweets(args.query, max_results=args.max_results)
    saved_path = save_dataframe(df, RAW_DATA_PATH)
    print(f"Saved {len(df)} tweets to {saved_path}")
