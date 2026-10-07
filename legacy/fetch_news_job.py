import os, datetime, requests, pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------- Load from Env Vars -----------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")

if not NEWS_API_KEY or not MODEL_PATH:
    raise ValueError("❌ Missing NEWS_API_KEY or MODEL_PATH env variables")

# ----------------- Load Model -----------------
print(f"🔄 Loading transformer model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=False)
model.eval()

id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
    return id2label[pred_id]

# ----------------- Fetch News -----------------
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}"

    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        print("❌ Error fetching news:", e)
        return

    articles = []
    for a in data.get("articles", []):
        title = a.get("title") or ""
        url_ = a.get("url") or ""
        text_to_analyze = title + " " + (a.get("description") or "")
        sentiment = predict_sentiment(text_to_analyze)

        if sentiment == "Positive":
            suggestion = "Stakeholders should consider BUY / expansion opportunities."
        elif sentiment == "Negative":
            suggestion = "Stakeholders should be cautious, consider SELL / risk mitigation."
        else:
            suggestion = "Neutral outlook, HOLD and monitor news flow."

        articles.append({
            "title": title,
            "content": a.get("content"),
            "source": a.get("source", {}).get("name"),
            "published": a.get("publishedAt"),
            "url": url_,
            "sentiment": sentiment,
            "suggestion": suggestion,
            "scraped_at": datetime.datetime.now().isoformat()
        })

    if not articles:
        print("⚠ No articles fetched")
        return

    df_new = pd.DataFrame(articles)
    out_path = os.path.join("data", "processed", "realtime", "data_all.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Merge with existing data
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["title", "url", "published"], keep="first", inplace=True)
    else:
        df_combined = df_new

    # Sort
    df_combined["published_dt"] = pd.to_datetime(df_combined["published"], errors="coerce")
    df_combined["scraped_dt"] = pd.to_datetime(df_combined["scraped_at"], errors="coerce")
    df_combined["sort_time"] = df_combined["published_dt"].fillna(df_combined["scraped_dt"])
    df_combined.sort_values(by="sort_time", ascending=False, inplace=True)

    df_combined.drop(columns=["published_dt", "scraped_dt", "sort_time"], inplace=True)
    df_combined.to_csv(out_path, index=False)

    print(f"[{datetime.datetime.now()}] ✅ Fetched {len(df_new)} news, total saved: {len(df_combined)}")

if __name__ == "__main__":
    fetch_news()