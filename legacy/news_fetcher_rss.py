import feedparser
import json
import os
from datetime import datetime
from urllib.parse import quote  # for URL encoding

# -----------------------------
# Encode query to handle spaces/special chars
# -----------------------------
query = "finance OR stock OR market"
encoded_query = quote(query)

rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

feed = feedparser.parse(rss_url)

# -----------------------------
# Extract articles
# -----------------------------
articles = []
for entry in feed.entries:
    articles.append({
        "title": entry.title,
        "link": entry.link,
        "published": entry.published
    })

# -----------------------------
# Save to raw/
# -----------------------------
out_dir = "data/raw/rss_news"
os.makedirs(out_dir, exist_ok=True)
filename = os.path.join(out_dir, datetime.now().strftime("%Y%m%d") + "_google.json")

with open(filename, "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2)

print(f"Saved {len(articles)} Google News headlines")
