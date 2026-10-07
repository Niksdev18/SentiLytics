"""News fetching module for NewsAPI.

Features:
- Header authentication via X-Api-Key (key never in URL query string)
- Configurable timeout and retries on 5xx errors
- Robust handling of 401 Unauthorized and 429 Rate Limited responses
- Title deduplication and " - Source" suffix stripping
- Caches raw JSON responses to data/raw/newsapi/ and falls back to latest cache
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import re
import time
import requests

from src.config import CompanyConfig

logger = logging.getLogger(__name__)


def strip_source_suffix(title: str) -> str:
    """Strip ' - Source' from the end of news headlines."""
    if not title:
        return ""
    # Matches " - Bloomberg", " - CNBC", " - Reuters", etc.
    return re.sub(r"\s+-\s+[^-]+$", "", title).strip()


def cache_news_response(cache_dir: Path, symbol: str, data: Dict[str, Any]) -> None:
    """Save raw response JSON to cache directory."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write latest copy
        latest_file = cache_dir / f"latest_{symbol}.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Write timestamped copy
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ts_file = cache_dir / f"{symbol}_{ts}.json"
        with open(ts_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("Failed to cache news for %s: %s", symbol, e)


def load_cached_news(cache_dir: Path, symbol: str) -> Optional[Dict[str, Any]]:
    """Load latest cached news JSON for a symbol."""
    latest_file = cache_dir / f"latest_{symbol}.json"
    if latest_file.exists():
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error reading cache file %s: %s", latest_file, e)
    return None


def fetch_company_news(
    company: CompanyConfig,
    api_key: Optional[str] = None,
    page_size: int = 5,
    cache_dir: Optional[Path] = None,
    timeout: float = 10.0,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Fetch recent news articles for a company from NewsAPI with caching and fallbacks.

    Args:
        company: CompanyConfig instance.
        api_key: NewsAPI key.
        page_size: Number of articles to fetch.
        cache_dir: Directory to cache raw JSON responses.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on 5xx errors.

    Returns:
        List of parsed article dictionaries.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "newsapi"

    url = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": api_key} if api_key else {}
    params = {
        "q": company.query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }

    data: Optional[Dict[str, Any]] = None

    if api_key:
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    cache_news_response(cache_dir, company.symbol, data)
                    break
                elif resp.status_code in (401, 429):
                    logger.warning(
                        "NewsAPI returned %s (%s) for %s. Using cached fallback.",
                        resp.status_code,
                        resp.text[:100],
                        company.name,
                    )
                    break
                elif resp.status_code >= 500:
                    logger.warning(
                        "NewsAPI 5xx server error (%s) on attempt %d for %s",
                        resp.status_code,
                        attempt + 1,
                        company.name,
                    )
                    if attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning("NewsAPI unexpected status %s: %s", resp.status_code, resp.text[:100])
                    break
            except (requests.RequestException, TimeoutError) as e:
                logger.warning("Network error fetching news for %s: %s", company.name, e)
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))

    # If live fetch was unavailable or failed, load from cache
    if data is None:
        data = load_cached_news(cache_dir, company.symbol)
        if data:
            logger.info("Loaded cached news for %s (%s)", company.name, company.symbol)

    if not data or "articles" not in data:
        return []

    articles = []
    seen_titles = set()

    for item in data.get("articles", []):
        raw_title = item.get("title") or ""
        clean_title = strip_source_suffix(raw_title)

        if not clean_title or clean_title == "[Removed]":
            continue

        normalized_key = clean_title.lower()
        if normalized_key in seen_titles:
            continue
        seen_titles.add(normalized_key)

        articles.append({
            "title": clean_title,
            "raw_title": raw_title,
            "description": item.get("description") or "",
            "url": item.get("url") or "#",
            "published_at": item.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
            "source": item.get("source", {}).get("name") or "News",
            "company": company.name,
            "symbol": company.symbol,
            "etf": company.etf,
        })

    return articles
