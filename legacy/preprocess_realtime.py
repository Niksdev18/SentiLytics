import json
import os
import pandas as pd
from src.preprocessing.preprocess import clean_text  # reuse your existing cleaning function

# Path to today's RSS feed JSON
rss_file = "data/raw/rss_news/20250930_google.json"

with open(rss_file, "r", encoding="utf-8") as f:
    articles = json.load(f)

# Convert to DataFrame for easier processing
df = pd.DataFrame(articles)
print("Loaded articles:", len(df))
# Apply cleaning
df["cleaned_text"] = df["title"].apply(clean_text)

# Optional: filter out empty text after cleaning
df = df[df["cleaned_text"].str.strip() != ""]
out_dir = "data/processed/realtime"
os.makedirs(out_dir, exist_ok=True)

filename = os.path.join(out_dir, "rss_today.csv")
df.to_csv(filename, index=False)
print(f"Saved {len(df)} cleaned articles to {filename}")
