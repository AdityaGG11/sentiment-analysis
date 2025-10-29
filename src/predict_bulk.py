# src/predict_bulk.py
"""
Load a saved HF model (hf_model by default), predict sentiment for each row in
a cleaned CSV (expects column 'clean_text' and either 'label' or 'auto_sentiment'),
save a CSV with predictions and create a matplotlib pie chart.

Usage:
python -m src.predict_bulk --csv data/tweets_clean.csv --model_dir hf_model --out data/analysis_hf.csv --chart outputs/hf_pie.png
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict_batch(texts, tokenizer, model, batch_size=16):
    preds = []
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()
            batch_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            batch_preds = np.argmax(batch_probs, axis=1)
            preds.extend(batch_preds.tolist())
            probs.extend(batch_probs.tolist())
    return preds, probs

def id2label_map(model):
    # map numeric idx -> label string
    cfg = model.config
    if hasattr(cfg, "id2label"):
        # keys may be strings
        id2label = {int(k): v for k, v in cfg.id2label.items()}
    else:
        # fallback: numeric->string
        id2label = {i: str(i) for i in range(cfg.num_labels)}
    return id2label

def make_pie(counts: pd.Series, outpath: str):
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    explode = [0.05 if lbl != 'neutral' else 0 for lbl in labels]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, explode=explode)
    ax.set_title("Sentiment distribution")
    ax.axis("equal")
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath

def main(args):
    df = pd.read_csv(args.csv)
    if 'clean_text' not in df.columns:
        raise ValueError("Input CSV must contain 'clean_text' column.")
    texts = df['clean_text'].astype(str).tolist()

    tokenizer, model = load_model(args.model_dir)
    preds_idx, probs = predict_batch(texts, tokenizer, model, batch_size=args.batch_size)
    id2label = id2label_map(model)
    preds_labels = [id2label[int(i)] for i in preds_idx]

    # attach predictions and max-prob
    df['hf_pred'] = preds_labels
    df['hf_confidence'] = [max(p) for p in probs]

    # write CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

    # counts (ensure order positive, negative, neutral if present)
    counts = df['hf_pred'].value_counts().reindex(['positive','negative','neutral']).fillna(0).astype(int)
    print("Sentiment counts:\n", counts.to_string())

    # pie chart
    chart_path = make_pie(counts, args.chart)
    print(f"Saved pie chart to {chart_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/tweets_clean.csv", help="Input cleaned CSV")
    parser.add_argument("--model_dir", default="hf_model", help="HF model directory")
    parser.add_argument("--out", default="data/analysis_hf.csv", help="Output CSV with predictions")
    parser.add_argument("--chart", default="outputs/hf_pie.png", help="Output pie chart PNG path")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
