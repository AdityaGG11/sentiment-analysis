# src/predict.py
"""
Simple prediction CLI using the HF model saved to hf_model/

Usage:
  python -m src.predict --text "your tweet text here" --model_dir hf_model
"""
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def predict(text: str, tokenizer, model, topk: int = 3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred_idx = int(np.argmax(probs).item())
        id2label = model.config.id2label
        pred_label = id2label[str(pred_idx)] if str(pred_idx) in id2label else id2label[pred_idx] if isinstance(pred_idx,int) and pred_idx in id2label else str(pred_idx)
    # build sorted probs
    class_items = []
    for i, p in enumerate(probs):
        label = id2label[str(i)] if str(i) in id2label else id2label[i]
        class_items.append((label, float(p)))
    class_items = sorted(class_items, key=lambda x: x[1], reverse=True)
    return pred_label, class_items

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text to classify")
    parser.add_argument("--model_dir", default="hf_model", help="Directory containing saved HF model/tokenizer")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    pred, probs = predict(args.text, tokenizer, model)
    print("Input:", args.text)
    print("Predicted:", pred)
    print("Probabilities:")
    for label, p in probs:
        print(f"  {label}: {p:.3f}")
