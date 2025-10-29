import pandas as pd
pd.set_option('display.max_colwidth', 200)
df = pd.read_csv('data/tweets_clean.csv')
print(df[['text', 'clean_text', 'auto_sentiment']].head(10))