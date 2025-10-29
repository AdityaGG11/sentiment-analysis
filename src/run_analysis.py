# src/run_analysis.py
"""
Run a topic-focused sentiment analysis pipeline:
1) collect tweets for a given keyword
2) clean them
3) assign an automatic sentiment (TextBlob fallback + VADER as option)
4) save a CSV with columns: text, clean_text, sentiment
5) print a table to terminal and save a pie chart (matplotlib)

Usage (from project root):
python -m src.run_analysis --query "music" --max_results 200 --out data/analysis_music.csv --chart outputs/music_pie.png

Notes:
- Requires TWITTER_BEARER_TOKEN in your .env or environment.
- Re-uses existing src/data_collect.py and src/preprocess.py functions.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# reuse your functions
from src.data_collect import collect_tweets
from src.preprocess import clean_text, auto_label_text
from src.utils import save_dataframe

# optional: use VADER as a second opinion for short text
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except Exception:
    VADER_AVAILABLE = False
    _vader = None

def vader_label(text):
    """Return a simple vader-based label ('positive','negative','neutral')."""
    if not VADER_AVAILABLE:
        return None
    score = _vader.polarity_scores(text)["compound"]
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

def ensemble_label(cleaned_text, original_text):
    """
    Produce a label using TextBlob (auto_label_text) and fallback/ensemble with VADER if available.
    Priority: TextBlob polarity (auto_label_text). If VADER disagrees and score magnitude is strong,
    prefer VADER for short social media texts.
    """
    tb_label = auto_label_text(cleaned_text)
    if not VADER_AVAILABLE:
        return tb_label
    vader_lbl = vader_label(original_text)
    # if both agree -> return
    if vader_lbl == tb_label:
        return tb_label
    # if they disagree, decide based on VADER compound magnitude
    comp = _vader.polarity_scores(original_text)["compound"]
    if abs(comp) >= 0.4:
        return vader_lbl
    return tb_label

def make_pie_chart(counts: pd.Series, outpath: str):
    """Save a simple pie chart of counts to outpath (PNG)."""
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    # ensure colors and explode for visual clarity
    explode = [0.05 if lbl != 'neutral' else 0 for lbl in labels]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, explode=explode)
    ax.set_title("Sentiment distribution")
    ax.axis("equal")
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath

def run(query: str, max_results: int, out_csv: str, chart_path: str, show_table: bool):
    # 1) collect
    print(f"Collecting up to {max_results} tweets for query: {query!r}")
    df = collect_tweets(query, max_results=max_results)
    if df is None or df.shape[0] == 0:
        print("No tweets collected. Exiting.")
        return

    # 2) clean + label
    print("Cleaning text and producing sentiment labels...")
    df["clean_text"] = df["text"].fillna("").apply(clean_text)
    # ensemble labeling (TextBlob + optional VADER)
    df["sentiment"] = df.apply(lambda r: ensemble_label(r["clean_text"], r["text"]), axis=1)

    # 3) keep only relevant columns for export
    out_df = df[["id", "created_at", "text", "clean_text", "sentiment"]].copy()
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    save_dataframe(out_df, out_csv)
    print(f"Saved analysis CSV to: {out_csv}")

    # 4) print a clean two-column table to terminal (tweet | sentiment)
    if show_table:
        pd.set_option("display.max_colwidth", 200)
        print("\nSample (text, sentiment):")
        print(out_df[["text", "sentiment"]].head(20).to_string(index=False))

    # 5) pie chart summary
    counts = out_df["sentiment"].value_counts().reindex(["positive","negative","neutral"]).fillna(0).astype(int)
    print("\nSentiment counts:")
    print(counts.to_string())
    os.makedirs(os.path.dirname(chart_path) or ".", exist_ok=True)
    saved_chart = make_pie_chart(counts, chart_path)
    print(f"Saved pie chart to: {saved_chart}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run topic-focused sentiment analysis and produce CSV + pie chart")
    parser.add_argument("--query", "-q", required=True, help="Search keyword / query for tweets (e.g. 'music')")
    parser.add_argument("--max_results", "-n", default=200, type=int, help="Maximum tweets to fetch (small first, e.g., 50)")
    parser.add_argument("--out", "-o", default="data/analysis.csv", help="Output CSV path")
    parser.add_argument("--chart", "-c", default="outputs/sentiment_pie.png", help="Output chart PNG path")
    parser.add_argument("--no-table", action="store_true", help="Do not print table sample to terminal")
    args = parser.parse_args()

    run(args.query, args.max_results, args.out, args.chart, not args.no_table)
