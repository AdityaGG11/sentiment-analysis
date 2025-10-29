# src/predict_merged.py
import joblib, os, re
from termcolor import colored
import pandas as pd, numpy as np

MODEL = joblib.load("models/logistic_merged.joblib")
VEC = joblib.load("models/tfidf_merged.joblib")

ABUSIVE = {"kill","murder","die","hate","destroy","attack","rape","suicide","bomb","shoot","kill yourself","fuck you"}

def contains_abusive(s):
    s = s.lower()
    for k in ABUSIVE:
        if k in s:
            return True
    return False

def predict_text(text):
    if contains_abusive(text):
        return "negative", {"override": 1.0}
    x = VEC.transform([text])
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(x)[0]
        classes = list(MODEL.classes_)
        pred = classes[int(probs.argmax())]
        prob_map = {c: float(f"{p:.3f}") for c,p in zip(classes, probs)}
        # neutral heuristic
        if probs.max() < 0.60 or (sorted(probs)[-1]-sorted(probs)[-2]) < 0.18:
            return "neutral", prob_map
        return pred, prob_map
    else:
        return MODEL.predict(x)[0], {}

if __name__ == "__main__":
    while True:
        s = input("\nEnter text (or exit): ").strip()
        if s.lower() in {"exit","quit"}:
            break
        lab, probs = predict_text(s)
        color = "yellow" if lab=="neutral" else ("green" if lab=="positive" else "red")
        print(colored(f"Predicted: {lab}", color))
        print("Probabilities:", probs)
