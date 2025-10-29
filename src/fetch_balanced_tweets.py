import tweepy
import pandas as pd
import os

# ========== CONFIGURE HERE ==========
# Make sure you have a Twitter API Bearer Token
BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN_HERE"

# These topics are chosen to yield balanced sentiments
SEARCH_TERMS = [
    "I love", "I hate", "amazing", "terrible",
    "disappointed", "so happy", "so sad", "great service",
    "worst experience", "excited for", "angry about"
]

OUTPUT_PATH = "data/tweets_raw_balanced.csv"
MAX_TWEETS_PER_TERM = 50  # small, lightweight for your laptop
# ===================================

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

os.makedirs("data", exist_ok=True)
tweets_list = []

for term in SEARCH_TERMS:
    print(f"🔍 Fetching tweets for: {term}")
    try:
        response = client.search_recent_tweets(
            query=f"{term} -is:retweet lang:en",
            tweet_fields=["id", "text", "created_at"],
            max_results=50
        )
        if response.data:
            for tweet in response.data:
                tweets_list.append({
                    "term": term,
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at
                })
    except Exception as e:
        print(f"⚠️ Error fetching {term}: {e}")

# Save results
df = pd.DataFrame(tweets_list)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
print(f"\n✅ Saved {len(df)} tweets to {OUTPUT_PATH}")
