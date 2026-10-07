# src/utils/market_sentiment.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

# -----------------------------
# Load today's sentiment data
# -----------------------------
sentiment_file = "data/processed/realtime/rss_today_with_sentiment.csv"
df_sent = pd.read_csv(sentiment_file)
print("Loaded sentiment data:", len(df_sent))

# -----------------------------
# Map numeric sentiment labels
# -----------------------------
# Assuming CSV has 2=negative, 1=positive, 0=neutral
numeric_map = {2: -1, 1: 1, 0: 0}
df_sent['sentiment_score'] = df_sent['sentiment'].map(numeric_map)

# Drop rows where mapping failed
df_sent = df_sent.dropna(subset=['sentiment_score'])

if df_sent.empty:
    print("No valid sentiment data found. Exiting.")
    exit()

# -----------------------------
# Aggregate sentiment per day
# -----------------------------
df_sent['published'] = pd.to_datetime(df_sent['published']).dt.date
daily_sentiment = df_sent.groupby('published')['sentiment_score'].mean().reset_index()
daily_sentiment.rename(columns={'sentiment_score': 'avg_sentiment'}, inplace=True)
print("Daily sentiment:\n", daily_sentiment)

# -----------------------------
# Fetch stock/market data
# -----------------------------
start_date = daily_sentiment['published'].min()
end_date   = daily_sentiment['published'].max()
symbol     = "TSLA"

tsla = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
tsla = tsla.reset_index()
tsla.columns = [col if not isinstance(col, tuple) else col[0] for col in tsla.columns]
tsla['Date'] = tsla['Date'].dt.date

# -----------------------------
# Merge sentiment with stock data
# -----------------------------
merged = pd.merge(daily_sentiment, tsla, left_on='published', right_on='Date', how='left')
merged = merged[['published', 'avg_sentiment', 'Open', 'High', 'Low', 'Close', 'Volume']]

# -----------------------------
# Save merged data
# -----------------------------
out_dir = "data/processed/realtime"
os.makedirs(out_dir, exist_ok=True)
merged_file = os.path.join(out_dir, f"market_sentiment_{datetime.now().strftime('%Y%m%d')}.csv")
merged.to_csv(merged_file, index=False)

print(f"Merged sentiment & stock data saved to {merged_file}")
print(merged.head())
