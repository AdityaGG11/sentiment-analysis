# Sentiment Analysis on Twitter — Starter Repo

## Summary
Starter pipeline for collecting tweets via Twitter API v2, preprocessing, training a TF-IDF + Logistic Regression baseline, evaluating, and running a Streamlit demo.

## Structure
- `.gitignore` - ignored files
- `requirements.txt` - Python deps
- `data/` - raw and cleaned CSVs (ignored)
- `models/` - saved model artifacts (ignored)
- `notebooks/` - EDA notebooks
- `src/` - source code (collection, preprocessing, training)
- `app/` - Streamlit demo

## Quick setup
1. Create a virtual env and activate:
   - mac/linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
2. Install:
   ```bash
   pip install -r requirements.txt