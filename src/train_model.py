# src/train_model.py
"""
Train a transformer classifier (DistilBERT by default) on your cleaned CSV.

This variant uses a minimal set of TrainingArguments so it is compatible
with older transformers installations on CPU systems.
Usage:
  python -m src.train_model --data data/tweets_clean.csv --output_dir hf_model --model distilbert-base-uncased --epochs 1 --batch_size 4
"""
import argparse
import os
import random
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict
import evaluate
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def prepare_hf_dataset(df: pd.DataFrame, text_col: str, label_col: str, test_size: float = 0.2, seed: int = 42):
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df[label_col].astype(str))
    label2id = {lab: int(i) for i, lab in enumerate(le.classes_)}
    id2label = {int(i): lab for i, lab in enumerate(le.classes_)}

    if len(df) < 5:
        train_df = df
        test_df = df
    else:
        try:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label_id'])
        except Exception:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    ds_train = Dataset.from_pandas(train_df[[text_col, 'label_id']].rename(columns={text_col: 'text', 'label_id': 'label'}))
    ds_test = Dataset.from_pandas(test_df[[text_col, 'label_id']].rename(columns={text_col: 'text', 'label_id': 'label'}))
    ds = DatasetDict({"train": ds_train, "test": ds_test})
    return ds, label2id, id2label, le

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": float(acc), "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1)}

def main(args):
    seed_everything(42)

    print("Loading data from:", args.data)
    df = pd.read_csv(args.data)
    text_col = "clean_text"
    label_col = "label" if "label" in df.columns else "auto_sentiment"
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns: '{text_col}' and '{label_col}'")

    print("Preparing dataset...")
    ds, label2id, id2label, le = prepare_hf_dataset(df, text_col, label_col, test_size=args.test_size)

    num_labels = len(label2id)
    print(f"Detected labels: {label2id} (num_labels={num_labels})")
    print(f"Train size: {len(ds['train'])}, Test size: {len(ds['test'])}")

    print("Loading tokenizer & model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels,
                                                               id2label=id2label, label2id=label2id)

    print("Tokenizing dataset...")
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer, max_length=args.max_length), batched=True)
    keep_cols = ["input_ids", "attention_mask", "label"]
    cols_to_remove = [c for c in tokenized["train"].column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(cols_to_remove)

    os.makedirs(args.output_dir, exist_ok=True)

    # Minimal TrainingArguments to maintain compatibility across transformers versions
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=max(1, args.epochs),
        per_device_train_batch_size=max(1, args.batch_size),
        per_device_eval_batch_size=max(1, args.batch_size),
        learning_rate=args.lr,
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete. Running evaluation on test set...")
    eval_results = trainer.evaluate(eval_dataset=tokenized["test"])
    print("Evaluation results:", eval_results)

    print("Saving model and tokenizer to", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model and tokenizer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to cleaned CSV (with clean_text and label/auto_sentiment)")
    parser.add_argument("--output_dir", default="hf_model", help="Output directory for saved model/tokenizer")
    parser.add_argument("--model", default="distilbert-base-uncased", help="HF model name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
