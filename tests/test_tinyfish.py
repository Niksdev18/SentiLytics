"""Unit tests for TinyFish integration and provider routing."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.config import AppConfig, CompanyConfig
from src.news import fetch_company_news, fetch_company_news_tinyfish, strip_source_suffix
from src.tinyfish import TinyFishClient


def test_tinyfish_client_is_configured():
    """Verify is_configured check based on API key presence."""
    client_empty = TinyFishClient(api_key="")
    assert client_empty.is_configured is False

    client_none = TinyFishClient(api_key=None)
    assert client_none.is_configured is False

    client_valid = TinyFishClient(api_key="sk-tinyfish-testkey123")
    assert client_valid.is_configured is True


def test_tinyfish_search_mocked(monkeypatch):
    """Verify search returns parsed results and caps at page_size."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "title": f"Apple Stock Headline {i} - Bloomberg",
                "snippet": f"Snippet for headline {i}",
                "url": f"https://example.com/apple/{i}",
                "publication_date": "2026-10-30T10:00:00Z",
                "site_name": "Bloomberg",
            }
            for i in range(10)
        ]
    }

    import requests
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)

    client = TinyFishClient(api_key="sk-tinyfish-test")
    res = client.search(query="Apple stock", page_size=3)

    assert "results" in res
    assert len(res["results"]) == 3
    assert res["results"][0]["site_name"] == "Bloomberg"


def test_fetch_company_news_tinyfish_parsing(tmp_path, monkeypatch):
    """Verify news articles are parsed and sanitized according to SentiLytics schema."""
    comp = CompanyConfig("Apple", "AAPL", "Apple stock", "XLK")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "title": "Apple Q4 Earnings Beat Expectations - Reuters",
                "snippet": "Apple beat analyst expectations across all metrics.",
                "url": "https://reuters.com/apple-q4",
                "publication_date": "2026-10-30T12:00:00Z",
                "site_name": "Reuters",
            },
            {
                # Duplicate headline after stripping suffix
                "title": "Apple Q4 Earnings Beat Expectations - CNBC",
                "snippet": "Duplicate headline should be skipped.",
                "url": "https://cnbc.com/apple-q4",
                "publication_date": "2026-10-30T12:05:00Z",
                "site_name": "CNBC",
            },
            {
                "title": "Apple expands supply chain partnerships - Bloomberg",
                "snippet": "New manufacturing lines planned.",
                "url": "https://bloomberg.com/supply-chain",
                "publication_date": "2026-10-30T14:00:00Z",
                "site_name": "Bloomberg",
            },
        ]
    }

    import requests
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)

    articles = fetch_company_news_tinyfish(
        company=comp,
        api_key="sk-tinyfish-test",
        page_size=5,
        cache_dir=tmp_path / "cache",
    )

    # 3 items returned by API, but 1 is duplicate -> 2 distinct articles
    assert len(articles) == 2

    # Suffix stripped
    assert articles[0]["title"] == "Apple Q4 Earnings Beat Expectations"
    assert articles[0]["source"] == "Reuters"
    assert articles[0]["company"] == "Apple"
    assert articles[0]["symbol"] == "AAPL"
    assert articles[0]["url"] == "https://reuters.com/apple-q4"
    assert "beat analyst expectations" in articles[0]["description"]

    # Verify cached file written
    cache_file = tmp_path / "cache" / "latest_AAPL.json"
    assert cache_file.exists()


def test_fetch_company_news_auto_provider_routing(monkeypatch):
    """Verify unified fetch_company_news routes to TinyFish when key is provided."""
    comp = CompanyConfig("Tesla", "TSLA", "Tesla stock", "CARZ")

    called_providers = []

    def fake_tinyfish(*args, **kwargs):
        called_providers.append("tinyfish")
        return [{
            "title": "Tesla robotaxi fleet begins pilot testing",
            "raw_title": "Tesla robotaxi fleet begins pilot testing",
            "description": "Autonomous testing begins in Texas.",
            "url": "https://example.com/tesla",
            "published_at": "2026-10-30T09:00:00Z",
            "source": "Electrek",
            "company": "Tesla",
            "symbol": "TSLA",
            "etf": "CARZ",
        }]

    def fake_newsapi(*args, **kwargs):
        called_providers.append("newsapi")
        return []

    with patch("src.news.fetch_company_news_tinyfish", side_effect=fake_tinyfish), \
         patch("src.news.fetch_company_news_newsapi", side_effect=fake_newsapi):

        articles = fetch_company_news(
            company=comp,
            api_key="newsapi_dummy_key",
            tinyfish_api_key="sk-tinyfish-valid_key",
            provider="auto",
        )

        assert "tinyfish" in called_providers
        assert "newsapi" not in called_providers
        assert len(articles) == 1
        assert articles[0]["symbol"] == "TSLA"


def test_tinyfish_fetch_content_mocked(monkeypatch):
    """Verify TinyFishClient.fetch_content handles batch markdown extraction."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "url": "https://example.com/article1",
                "format": "markdown",
                "text": "# Article 1\nFull text content here.",
            }
        ],
        "errors": [],
    }

    import requests
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)

    client = TinyFishClient(api_key="sk-tinyfish-test")
    res = client.fetch_content(["https://example.com/article1"])

    assert len(res["results"]) == 1
    assert "Full text content" in res["results"][0]["text"]
