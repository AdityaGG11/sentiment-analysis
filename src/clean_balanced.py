import pandas as pd
import re
import os

# Define cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# Load dataset
input_path = os.path.join("data", "tweets_raw_balanced.csv")
output_path = os.path.join("data", "tweets_clean_balanced.csv")

df = pd.read_csv(input_path)

# Clean text column
df["clean_text"] = df["text"].apply(clean_text)

# Save the cleaned dataset
df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved to: {output_path}")
print(df.head())
