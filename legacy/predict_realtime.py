import pandas as pd
import requests, os, datetime, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------
# Load Model
# -----------------
MODEL_PATH = os.path.join("models", "transformer_model", "best_model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
    return id2label[pred_id]

# -----------------
# Fetch News (Example: NewsAPI or RSS)
# -----------------
def fetch_news():
    API_KEY = "YOUR_API_KEY"   # replace with real API key
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={API_KEY}"
    data = requests.get(url).json()

    articles = []
    for a in data.get("articles", []):
        text_to_analyze = (a.get("title") or "") + " " + (a.get("description") or "")
        sentiment = predict_sentiment(text_to_analyze)

        if sentiment == "Positive":
            suggestion = "Stakeholders should consider BUY / expansion opportunities."
        elif sentiment == "Negative":
            suggestion = "Stakeholders should be cautious, consider SELL / risk mitigation."
        else:
            suggestion = "Neutral outlook, HOLD and monitor news flow."

        articles.append({
            "title": a.get("title"),
            "content": a.get("content"),
            "source": a.get("source", {}).get("name"),
            "published": a.get("publishedAt"),
            "url": a.get("url"),
            "sentiment": sentiment,
            "suggestion": suggestion,
            "scraped_at": datetime.datetime.now().isoformat()
        })

    if not articles:
        print("⚠️ No articles fetched")
        return

    df = pd.DataFrame(articles)

    # One single CSV for all time
    out_path = os.path.join("data", "processed", "realtime", "data_all.csv")

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    # Append to the same file (keep all history)
    df.to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))

    print(f"[{datetime.datetime.now()}] Added {len(df)} news to {out_path}")

# -----------------
# Run every 30 minutes
# -----------------
if __name__ == "__main__":
    while True:
        fetch_news()
        time.sleep(1800)  # 30 min
