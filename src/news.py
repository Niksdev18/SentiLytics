"""News fetching module supporting TinyFish and NewsAPI providers.

Features:
- Dual-provider support: TinyFish (search & web agent) and NewsAPI
- Header authentication via X-API-Key (key never in URL query string)
- Configurable timeout and retries on 5xx errors
- Robust handling of 401 Unauthorized and 429 Rate Limited responses
- Title deduplication and " - Source" suffix stripping
- Caches raw JSON responses to data/raw/tinyfish/ and data/raw/newsapi/
- Automatic fallback from TinyFish -> NewsAPI -> Local Cache
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import logging
import re
import time
import requests

from src.config import CompanyConfig
from src.tinyfish import TinyFishClient

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


def fetch_company_news_tinyfish(
    company: CompanyConfig,
    api_key: Optional[str] = None,
    page_size: int = 5,
    cache_dir: Optional[Path] = None,
    timeout: float = 10.0,
    max_retries: int = 2,
    client: Optional[TinyFishClient] = None,
) -> List[Dict[str, Any]]:
    """Fetch company news using TinyFish Search API.

    Args:
        company: CompanyConfig instance.
        api_key: TinyFish API key.
        page_size: Maximum number of articles to return.
        cache_dir: Directory for storing raw cache JSON files.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on network/5xx errors.
        client: Optional pre-configured TinyFishClient.

    Returns:
        List of parsed article dictionaries adhering to SentiLytics schema.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "tinyfish"

    tf_client = client or TinyFishClient(api_key=api_key, timeout=timeout, max_retries=max_retries)
    data: Optional[Dict[str, Any]] = None

    if tf_client.is_configured:
        # Build natural search query for stock news
        query = f"{company.name} stock news"
        search_resp = tf_client.search(query=query, page_size=page_size)
        if search_resp and search_resp.get("results"):
            data = search_resp
            cache_news_response(cache_dir, company.symbol, data)

    # Fallback to cached TinyFish responses if live fetch returned nothing
    if data is None:
        data = load_cached_news(cache_dir, company.symbol)
        if data:
            logger.info("Loaded cached TinyFish news for %s (%s)", company.name, company.symbol)

    if not data:
        return []

    items = data.get("results") or data.get("articles") or []
    if isinstance(items, dict):
        items = [items]

    articles: List[Dict[str, Any]] = []
    seen_titles = set()

    for item in items:
        raw_title = item.get("title") or item.get("headline") or ""
        clean_title = strip_source_suffix(raw_title)

        if not clean_title or clean_title == "[Removed]":
            continue

        normalized_key = clean_title.lower()
        if normalized_key in seen_titles:
            continue
        seen_titles.add(normalized_key)

        source_name = (
            item.get("site_name")
            or (item.get("source", {}).get("name") if isinstance(item.get("source"), dict) else item.get("source"))
            or "TinyFish"
        )

        articles.append({
            "title": clean_title,
            "raw_title": raw_title,
            "description": item.get("snippet") or item.get("description") or "",
            "url": item.get("url") or item.get("link") or "#",
            "published_at": item.get("publication_date") or item.get("publishedAt") or datetime.now(timezone.utc).isoformat(),
            "source": source_name,
            "company": company.name,
            "symbol": company.symbol,
            "etf": company.etf,
        })

    return articles[:page_size]


def fetch_company_news_newsapi(
    company: CompanyConfig,
    api_key: Optional[str] = None,
    page_size: int = 5,
    cache_dir: Optional[Path] = None,
    timeout: float = 10.0,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Fetch recent news articles for a company from NewsAPI with caching and fallbacks."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "newsapi"

    url = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": api_key} if api_key else {}
    params: Dict[str, Union[str, int]] = {
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


def fetch_company_news(
    company: CompanyConfig,
    api_key: Optional[str] = None,
    page_size: int = 5,
    cache_dir: Optional[Path] = None,
    timeout: float = 10.0,
    max_retries: int = 2,
    tinyfish_api_key: Optional[str] = None,
    provider: str = "auto",
) -> List[Dict[str, Any]]:
    """Unified news fetcher supporting TinyFish and NewsAPI with automatic fallback.

    Args:
        company: CompanyConfig instance.
        api_key: NewsAPI key.
        page_size: Number of articles to fetch.
        cache_dir: Directory to cache raw JSON responses.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on 5xx errors.
        tinyfish_api_key: TinyFish API key.
        provider: 'auto', 'tinyfish', or 'newsapi'.

    Returns:
        List of parsed article dictionaries.
    """
    provider_clean = (provider or "auto").strip().lower()

    # 1. Try TinyFish if provider is 'tinyfish' or 'auto' (with TinyFish key available)
    if provider_clean in ("tinyfish", "auto") and tinyfish_api_key:
        logger.info("Fetching news for %s via TinyFish...", company.name)
        articles = fetch_company_news_tinyfish(
            company=company,
            api_key=tinyfish_api_key,
            page_size=page_size,
            timeout=timeout,
            max_retries=max_retries,
        )
        if articles:
            return articles
        if provider_clean == "tinyfish":
            logger.warning("TinyFish fetch returned no results for %s.", company.name)
            return []

    # 2. Try NewsAPI if provider is 'newsapi' or 'auto' (fallback or explicit)
    if provider_clean in ("newsapi", "auto") and api_key:
        logger.info("Fetching news for %s via NewsAPI...", company.name)
        articles = fetch_company_news_newsapi(
            company=company,
            api_key=api_key,
            page_size=page_size,
            cache_dir=cache_dir,
            timeout=timeout,
            max_retries=max_retries,
        )
        if articles:
            return articles

    # 3. Check cached data from TinyFish cache then NewsAPI cache
    root = Path(__file__).resolve().parent.parent
    for sub in ("tinyfish", "newsapi"):
        candidate_cache = root / "data" / "raw" / sub
        cached = load_cached_news(candidate_cache, company.symbol)
        if cached:
            if "results" in cached:
                # Parse tinyfish cache format
                return fetch_company_news_tinyfish(
                    company=company,
                    api_key=None,
                    page_size=page_size,
                    cache_dir=candidate_cache,
                )
            elif "articles" in cached:
                # Parse newsapi cache format
                return fetch_company_news_newsapi(
                    company=company,
                    api_key=None,
                    page_size=page_size,
                    cache_dir=candidate_cache,
                )

    return []
