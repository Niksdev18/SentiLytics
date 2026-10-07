from flask import Flask, render_template, jsonify
import pandas as pd
import os
import datetime
import requests
import yaml
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from apscheduler.schedulers.background import BackgroundScheduler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ----------------- Load Config -----------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

NEWS_API_KEY = config["api_keys"]["newsapi"]
MODEL_PATH = config["model"]["path"]

app = Flask(__name__)

# ----------------- Load Transformer Model -----------------
print(f"🔄 Loading model from: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

model.to(device)
model.eval()

id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ----------------- Load Model Metrics -----------------
METRICS_PATH = os.path.join(MODEL_PATH, "metrics.json")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        MODEL_METRICS = json.load(f)
else:
    MODEL_METRICS = {}

# ----------------- Sentiment Analyzer -----------------
vader = SentimentIntensityAnalyzer()

# ----------------- Company List -----------------
COMPANIES = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Meta": "META",
    "NVIDIA": "NVDA",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

# ----------------- Sentiment Prediction -----------------
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    neg_prob, neu_prob, pos_prob = probs
    base_pred = id2label[int(probs.argmax())]

    # avoid over-neutral prediction
    if base_pred == "Neutral" and neu_prob < 0.70:
        base_pred = "Positive" if pos_prob > neg_prob else "Negative"

    # VADER check
    compound = vader.polarity_scores(text)["compound"]

    if base_pred == "Neutral":
        if compound >= 0.05:
            base_pred = "Positive"
        elif compound <= -0.05:
            base_pred = "Negative"

    confidence = round(max(neg_prob, neu_prob, pos_prob) * 100, 2)

    return {
        "sentiment": base_pred,
        "confidence": confidence
    }

# ----------------- Company Aggregated Sentiment -----------------
def aggregate_company_sentiment(articles):

    sentiments = []

    for article in articles:

        text = (article.get("title") or "") + " " + (article.get("description") or "")

        if text.strip():
            sentiments.append(predict_sentiment(text))

    if not sentiments:
        return {"overall_sentiment": "Neutral", "confidence": 0}

    positive = sum(1 for s in sentiments if s["sentiment"] == "Positive")
    negative = sum(1 for s in sentiments if s["sentiment"] == "Negative")
    neutral = sum(1 for s in sentiments if s["sentiment"] == "Neutral")

    if positive > negative and positive > neutral:
        overall = "Positive"
    elif negative > positive and negative > neutral:
        overall = "Negative"
    else:
        overall = "Neutral"

    avg_conf = sum(s["confidence"] for s in sentiments) / len(sentiments)

    return {
        "overall_sentiment": overall,
        "confidence": round(avg_conf, 2)
    }

# ----------------- Stock Data -----------------
def get_stock_data(symbol):

    try:

        stock = yf.Ticker(symbol)

        hist = stock.history(period="5d")

        if hist.empty:
            return {"price": 0, "change": 0}

        before = hist["Close"].iloc[-2]
        after = hist["Close"].iloc[-1]

        change = ((after - before) / before) * 100

        return {
            "price": round(after, 2),
            "change": round(change, 2)
        }

    except:
        return {"price": 0, "change": 0}

# ----------------- Trading Suggestion -----------------
def generate_trading_suggestion(sentiment, change):

    if sentiment == "Positive" and change > 1:
        return "🟢 BUY"

    if sentiment == "Negative" and change < -1:
        return "🔴 SELL"

    return "⚪ HOLD"

# ----------------- News Fetcher -----------------
def fetch_news():

    all_news = []

    for company, symbol in COMPANIES.items():

        print(f"📡 Fetching: {company}")

        url = f"https://newsapi.org/v2/everything?q={company}&language=en&pageSize=5&sortBy=publishedAt&apiKey={NEWS_API_KEY}"

        try:
            response = requests.get(url).json()
        except:
            continue

        articles = response.get("articles", [])

        sentiment = aggregate_company_sentiment(articles)

        stock = get_stock_data(symbol)

        suggestion = generate_trading_suggestion(
            sentiment["overall_sentiment"],
            stock["change"]
        )

        for a in articles:

            title = a.get("title") or ""
            desc = a.get("description") or ""

            text = title + " " + desc

            article_sent = predict_sentiment(text)

            all_news.append({

                "company": company,
                "symbol": symbol,
                "title": title,
                "url": a.get("url"),
                "source": a.get("source", {}).get("name"),
                "published": a.get("publishedAt"),

                "article_sentiment": article_sent["sentiment"],
                "article_confidence": article_sent["confidence"],

                "overall_sentiment": sentiment["overall_sentiment"],
                "overall_confidence": sentiment["confidence"],

                "stock_price": stock["price"],
                "price_change": stock["change"],

                "suggestion": suggestion,

                "scraped_at": datetime.datetime.now().isoformat()

            })

    df = pd.DataFrame(all_news)

    os.makedirs("data/processed/realtime", exist_ok=True)

    df.to_csv("data/processed/realtime/company_news.csv", index=False)

    print(f"✅ Saved {len(df)} articles")

# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():

    path = "data/processed/realtime/company_news.csv"

    if not os.path.exists(path):
        return render_template("dashboard.html", news=[], metrics=None, model_metrics=MODEL_METRICS)

    df = pd.read_csv(path)

    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")

    df.sort_values(by="published_dt", ascending=False, inplace=True)

    news = df.head(30).to_dict(orient="records")

    return render_template(
        "dashboard.html",
        news=news,
        metrics=None,
        model_metrics=MODEL_METRICS
    )

@app.route("/api/model_metrics")
def api_model_metrics():
    return jsonify(MODEL_METRICS)

# ----------------- Scheduler -----------------
if __name__ == "__main__":

    scheduler = BackgroundScheduler()

    scheduler.add_job(fetch_news, "interval", hours=2)

    scheduler.start()

    print("🚀 Initial news fetch...")

    fetch_news()

    print("🌐 Running Flask server http://127.0.0.1:5000")

    app.run(debug=True, port=5000, use_reloader=False)