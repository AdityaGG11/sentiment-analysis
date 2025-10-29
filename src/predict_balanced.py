# src/predict_balanced.py
import joblib, re, math
from pathlib import Path
from termcolor import colored

# Load artifacts (must be the ones saved during training)
MODEL_PATH = Path("models/sentiment_model.pkl")
VEC_PATH = Path("models/vectorizer.pkl")

if not MODEL_PATH.exists() or not VEC_PATH.exists():
    raise SystemExit("Missing model or vectorizer in models/. Train first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# abusive/violent keywords -> force negative
ABUSIVE_KEYWORDS = {
    "kill","murder","die","hate","destroy","attack","rape","suicide","bomb","shoot","kill yourself",
    "fuck you","die", "i will kill", "i want to kill"
}
# profanity tokens that strongly signal negative sentiment
PROFANITY = {"fuck","shit","bastard","bitch","asshole"}

def contains_phrase_set(text, phrases):
    t = text.lower()
    for p in phrases:
        if p in t:
            return True
    return False

def neutral_by_rule(text, probs, classes):
    # If model has 3 classes and includes 'neutral', treat low-confidence as neutral
    maxp = max(probs)
    diff = None
    sorted_probs = sorted(probs, reverse=True)
    if len(sorted_probs) >= 2:
        diff = sorted_probs[0] - sorted_probs[1]
    else:
        diff = sorted_probs[0]

    # rule 1: very low max probability -> neutral
    if maxp < 0.60:
        return True

    # rule 2: model is unsure between top classes
    if diff is not None and diff < 0.18:
        return True

    # rule 3: short non-opinion phrases (single word like "hello") -> neutral
    if len(text.split()) <= 2 and maxp < 0.85:
        return True

    return False

def colored_print(sentiment, probs, classes):
    col = "yellow"
    if sentiment == "positive":
        col = "green"
    elif sentiment == "negative":
        col = "red"
    print(colored(f"\nPredicted Sentiment: {sentiment}", col))
    # print probabilities for debugging, in class order
    prob_map = {c: float(f"{p:.3f}") for c,p in zip(classes, probs)}
    print("Probabilities:", prob_map)

print("\nType sentences (or 'exit' to quit). This predictor applies safety overrides and neutral heuristics.\n")

while True:
    text = input("Enter text: ").strip()
    if text.lower() in {"exit", "quit"}:
        break
    if not text:
        print(colored("Please enter a sentence.", "yellow"))
        continue

    # 1) Rule-based override for violent / abusive phrases
    if contains_phrase_set(text, ABUSIVE_KEYWORDS) or contains_phrase_set(text, PROFANITY):
        colored_print("negative", [1.0], ["negative"])
        continue

    # 2) Transform with saved vectorizer and get model probabilities
    X = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
    else:
        # fallback: only predict, no probs
        pred = model.predict(X)[0]
        colored_print(pred, [1.0], [pred])
        continue

    # 3) Map classes & compute neutral heuristics
    # Ensure classes are strings like 'positive','negative','neutral'
    # Determine top predicted label according to model
    top_idx = int(probs.argmax())
    top_label = classes[top_idx]

    # If model already includes 'neutral' as a class, respect it but still check confidence
    if "neutral" in classes:
        # If model predicts neutral directly, accept it.
        if top_label == "neutral":
            colored_print("neutral", probs, classes)
            continue
        # else, if low confidence, override to neutral
        if neutral_by_rule(text, probs, classes):
            colored_print("neutral", probs, classes)
            continue
        # otherwise accept model's prediction
        colored_print(top_label, probs, classes)
        continue
    else:
        # model is binary (positive/negative) — use thresholds + neutral heuristic
        maxp = float(probs.max())
        # if overall low confidence or close probs => neutral
        if neutral_by_rule(text, probs, classes):
            colored_print("neutral", probs, classes)
            continue
        # otherwise accept top_label
        colored_print(top_label, probs, classes)
        continue
