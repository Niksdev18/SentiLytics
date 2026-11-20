"""Comprehensive test suite for Phase 2: Backend Core."""

from datetime import datetime, timezone, timedelta
import pandas as pd

from src.config import AppConfig, CompanyConfig
from src.demo import generate_demo_results
from src.insight import generate_trading_suggestion, is_correct, aggregate_accuracy
from src.news import strip_source_suffix
from src.prices import reaction_from_history, fetch_symbol_history
from src.pipeline import run_pipeline


# ─────────────────────────────────────────────────────────────
# 1. Suggestion Matrix & Correctness Scoring Tests
# ─────────────────────────────────────────────────────────────
def test_suggestion_matrix_all_branches():
    """Verify all 5 trading suggestion outcomes."""
    thresh = 0.5

    # Positive + up -> Consider Buy
    assert generate_trading_suggestion("Positive", 1.25, thresh, True) == "Consider Buy"
    # Negative + down -> Consider Sell
    assert generate_trading_suggestion("Negative", -1.25, thresh, True) == "Consider Sell"
    # Positive + down -> Watch
    assert generate_trading_suggestion("Positive", -1.25, thresh, True) == "Watch - possible delayed reaction"
    # Negative + up -> Watch
    assert generate_trading_suggestion("Negative", 1.25, thresh, True) == "Watch - possible delayed reaction"
    # Flat moves or Neutral/Mixed -> Hold
    assert generate_trading_suggestion("Positive", 0.20, thresh, True) == "Hold"
    assert generate_trading_suggestion("Negative", -0.10, thresh, True) == "Hold"
    assert generate_trading_suggestion("Neutral", 2.50, thresh, True) == "Hold"
    assert generate_trading_suggestion("Mixed", 2.50, thresh, True) == "Hold"
    # No price data -> Hold - no price data
    assert generate_trading_suggestion("Positive", None, thresh, False) == "Hold - no price data"
    assert generate_trading_suggestion("Positive", 1.5, thresh, False) == "Hold - no price data"


def test_is_correct_matrix():
    """Verify market direction correctness rules."""
    thresh = 0.5

    # Positive matches
    assert is_correct("Positive", 1.2, thresh, True) is True
    assert is_correct("Positive", -1.2, thresh, True) is False
    assert is_correct("Positive", 0.1, thresh, True) is False

    # Negative matches
    assert is_correct("Negative", -1.2, thresh, True) is True
    assert is_correct("Negative", 1.2, thresh, True) is False
    assert is_correct("Negative", 0.1, thresh, True) is False

    # Neutral matches (flat)
    assert is_correct("Neutral", 0.2, thresh, True) is True
    assert is_correct("Neutral", -0.3, thresh, True) is True
    assert is_correct("Neutral", 1.5, thresh, True) is False
    assert is_correct("Neutral", -1.5, thresh, True) is False

    # Excluded cases
    assert is_correct("Mixed", 1.5, thresh, True) is None
    assert is_correct("Positive", None, thresh, False) is None


def test_aggregate_accuracy():
    """Verify overall and per-company accuracy aggregation."""
    rows = [
        {"company": "Apple", "correct": True},
        {"company": "Apple", "correct": False},
        {"company": "Tesla", "correct": True},
        {"company": "Tesla", "correct": None},  # excluded
    ]
    res = aggregate_accuracy(rows)
    assert res["total"] == 3
    assert res["correct"] == 2
    assert res["accuracy_pct"] == 66.7
    assert res["mismatches"] == 1
    assert res["by_company"]["Apple"]["total"] == 2
    assert res["by_company"]["Apple"]["correct"] == 1
    assert res["by_company"]["Tesla"]["total"] == 1
    assert res["by_company"]["Tesla"]["correct"] == 1


# ─────────────────────────────────────────────────────────────
# 2. Price Edge Cases Tests
# ─────────────────────────────────────────────────────────────
def test_price_reaction_hourly_and_tz_handling():
    """Verify price reaction calculations with tz-aware hourly bars."""
    base = datetime(2026, 10, 5, 14, 0, tzinfo=timezone.utc)
    dates = [base + timedelta(hours=i) for i in range(10)]
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    hist = pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))

    # Published at +1 hour (15:00 UTC, Close=101.0)
    pub = base + timedelta(hours=1)
    # Window of 4 hours -> target is 19:00 UTC (Close=105.0)
    res = reaction_from_history(hist, pub, window_hours=4)

    assert res is not None
    assert res["before_close"] == 101.0
    assert res["after_close"] == 105.0
    # % change = (105 - 101) / 101 * 100 = 3.96%
    assert res["change_pct"] == 3.96
    assert len(res["graph_closes"]) > 0


def test_price_reaction_weekend_and_naive_index():
    """Verify weekend publication correctly falls back to nearest market bars with naive timestamps."""
    # Market closes Friday 2026-10-02 20:00, reopens Monday 2026-10-05 13:30
    dates = [
        datetime(2026, 10, 2, 19, 0),
        datetime(2026, 10, 2, 20, 0),
        datetime(2026, 10, 5, 14, 0),
        datetime(2026, 10, 5, 18, 0),
    ]
    hist = pd.DataFrame({"Close": [150.0, 152.0, 155.0, 160.0]}, index=pd.DatetimeIndex(dates))

    # Published on Saturday 2026-10-03 (naive string)
    pub_str = "2026-10-03 12:00:00"
    res = reaction_from_history(hist, pub_str, window_hours=4)

    assert res is not None
    assert res["before_close"] == 152.0  # Friday close
    assert res["after_close"] == 155.0   # Monday reaction bar (first bar after weekend)
    # % change = (155 - 152) / 152 * 100 = 1.97%
    assert res["change_pct"] == 1.97


def test_price_reaction_empty_or_missing_data():
    """Verify graceful handling when price data is empty or invalid."""
    empty_df = pd.DataFrame()
    assert reaction_from_history(empty_df, datetime.now(timezone.utc)) is None

    no_close_df = pd.DataFrame({"Volume": [100]}, index=[datetime.now(timezone.utc)])
    assert reaction_from_history(no_close_df, datetime.now(timezone.utc)) is None


def test_etf_fallback_mechanism(monkeypatch):
    """Verify fallback to sector ETF when ticker data is empty."""
    def fake_ticker(sym):
        class Dummy:
            def history(self, period="5d", interval="1h"):
                if sym == "AAPL":
                    return pd.DataFrame()  # empty ticker
                elif sym == "XLK":
                    idx = pd.date_range("2026-10-01", periods=2, freq="h")
                    return pd.DataFrame({"Close": [200.0, 202.0]}, index=idx)
                return pd.DataFrame()
        return Dummy()

    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", fake_ticker)

    hist, proxy = fetch_symbol_history("AAPL", proxy_etf="XLK")
    assert not hist.empty
    assert proxy == "XLK"


# ─────────────────────────────────────────────────────────────
# 3. News Processing Tests
# ─────────────────────────────────────────────────────────────
def test_strip_source_suffix():
    """Verify source suffixes are stripped from headlines."""
    assert strip_source_suffix("Apple reports Q4 earnings beat - Reuters") == "Apple reports Q4 earnings beat"
    assert strip_source_suffix("Tesla expands Gigafactory - Bloomberg") == "Tesla expands Gigafactory"
    assert strip_source_suffix("Headline without suffix") == "Headline without suffix"


# ─────────────────────────────────────────────────────────────
# 4. Pipeline with Stubbed Dependencies
# ─────────────────────────────────────────────────────────────
def test_run_pipeline_end_to_end_with_stubs(tmp_path):
    """Verify end-to-end pipeline execution with stubbed news and prices."""
    cfg = AppConfig(
        companies=[CompanyConfig("Apple", "AAPL", "Apple stock", "XLK")],
        demo_mode=False,
    )
    cfg.root_dir = tmp_path

    # Stubbed news
    def stub_news(comp, key, page_size):
        return [{
            "title": "Apple launches new M4 chip with major speedups",
            "url": "https://example.com/apple",
            "source": "TechWire",
            "published_at": "2026-10-05T12:00:00Z",
            "company": comp.name,
            "symbol": comp.symbol,
            "etf": comp.etf,
        }]

    # Stubbed price history
    def stub_prices(sym, etf):
        dates = [
            datetime(2026, 10, 5, 11, 0, tzinfo=timezone.utc),
            datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 10, 5, 16, 0, tzinfo=timezone.utc),
        ]
        df = pd.DataFrame({"Close": [100.0, 100.0, 105.0]}, index=pd.DatetimeIndex(dates))
        return df, None

    # Stubbed sentiment engine
    class StubEngine:
        model_description = "Stubbed FinBERT"

        def predict_batch(self, texts):
            return [{
                "label": "Positive",
                "confidence": 0.95,
                "probabilities": {"Positive": 0.95, "Neutral": 0.03, "Negative": 0.02},
            }]

    results = run_pipeline(
        config=cfg,
        engine=StubEngine(),
        news_fn=stub_news,
        history_fn=stub_prices,
    )

    assert results["source"] == "live"
    assert len(results["rows"]) == 1
    row = results["rows"][0]

    assert row["company"] == "Apple"
    assert row["sentiment"] == "Positive"
    assert row["change_pct"] == 5.0
    assert row["suggestion"] == "Consider Buy"
    assert row["correct"] is True
    assert row["price_available"] is True

    # Verify results files were written
    json_path = tmp_path / "data" / "processed" / "realtime" / "results.json"
    csv_path = tmp_path / "data" / "processed" / "realtime" / "results.csv"
    assert json_path.exists()
    assert csv_path.exists()


# ─────────────────────────────────────────────────────────────
# 5. Demo Data Generator Tests
# ─────────────────────────────────────────────────────────────
def test_demo_mode_output():
    """Verify deterministic demo data generation."""
    cfg = AppConfig()
    demo_data = generate_demo_results(cfg)

    assert demo_data["source"] == "demo"
    assert len(demo_data["rows"]) > 0

    # Ensure every single headline is explicitly tagged as demo
    for r in demo_data["rows"]:
        assert r["title"].startswith("[DEMO]"), f"Headline not marked demo: {r['title']}"
        assert r["is_demo"] is True

    # Ensure all companies exist
    companies_in_demo = {r["company"] for r in demo_data["rows"]}
    for comp in cfg.companies:
        assert comp.name in companies_in_demo

    # Ensure metrics exist and are non-empty
    assert "accuracy_pct" in demo_data["metrics"]
    assert "by_company" in demo_data["metrics"]
