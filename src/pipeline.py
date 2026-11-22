"""Main data pipeline coordinating news fetching, sentiment scoring, and price reaction analysis.

Features:
- Pure injectable dependencies for comprehensive unit testing
- Single-pass batch sentiment scoring (scores each headline exactly once)
- Price reaction tracking around publish time with ETF fallback
- Outputs data/processed/realtime/results.json and CSV export
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Tuple
import json
import logging
import pandas as pd

from src.config import AppConfig, CompanyConfig, load_config
from src.demo import generate_demo_results
from src.insight import generate_trading_suggestion, is_correct, aggregate_accuracy
from src.news import fetch_company_news
from src.prices import fetch_symbol_history, reaction_from_history
from src.sentiment import SentimentEngine

logger = logging.getLogger(__name__)


def run_pipeline(
    config: Optional[AppConfig] = None,
    engine: Optional[SentimentEngine] = None,
    news_fn: Optional[Callable[[CompanyConfig, Optional[str], int], List[Dict[str, Any]]]] = None,
    history_fn: Optional[Callable[[str, Optional[str]], Tuple[pd.DataFrame, Optional[str]]]] = None,
    force_demo: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end sentiment and market reaction pipeline.

    Args:
        config: Application configuration.
        engine: SentimentEngine instance (lazy loaded if None).
        news_fn: Callable to fetch company news (injectable for testing).
        history_fn: Callable to fetch price history (injectable for testing).
        force_demo: If True, generate synthetic demo data.

    Returns:
        Dict adhering to results.json schema.
    """
    cfg = config or load_config()

    if force_demo or cfg.demo_mode:
        logger.info("Executing pipeline in DEMO MODE.")
        results = generate_demo_results(cfg)
        _save_results(cfg, results)
        return results

    if news_fn is None:
        def _default_news_fn(comp: CompanyConfig, key: Optional[str], ps: int) -> List[Dict[str, Any]]:
            return fetch_company_news(
                comp,
                api_key=key,
                page_size=ps,
                tinyfish_api_key=cfg.tinyfish_api_key,
                provider=cfg.news_provider,
            )
        news_fn = _default_news_fn

    if history_fn is None:
        def _default_history_fn(sym: str, etf: str) -> Any:
            return fetch_symbol_history(sym, proxy_etf=etf)
        history_fn = _default_history_fn

    # 1. Fetch news articles across all companies
    all_articles: List[Dict[str, Any]] = []
    data_source = "live"

    for comp in cfg.companies:
        try:
            articles = news_fn(comp, cfg.newsapi_key, cfg.articles_per_company)
            all_articles.extend(articles)
        except Exception as e:
            logger.warning("Error fetching news for %s: %s", comp.name, e)

    # If no live news articles could be fetched (e.g., offline or no API key)
    if not all_articles:
        logger.warning("No news articles retrieved. Falling back to DEMO MODE.")
        demo_res = generate_demo_results(cfg)
        _save_results(cfg, demo_res)
        return demo_res

    # 2. Batch sentiment inference (score headlines ONCE)
    if engine is None:
        engine = SentimentEngine(cfg)
    titles = [a["title"] for a in all_articles]
    sentiment_outputs = engine.predict_batch(titles)

    # 3. Batch price histories per unique symbol
    unique_symbols = {a["symbol"]: a.get("etf", "SPY") for a in all_articles}
    histories: Dict[str, Tuple[pd.DataFrame, Optional[str]]] = {}

    for sym, etf in unique_symbols.items():
        try:
            histories[sym] = history_fn(sym, etf)
        except Exception as e:
            logger.warning("Failed to fetch price history for %s: %s", sym, e)
            histories[sym] = (pd.DataFrame(), None)

    # 4. Correlate news sentiment with price reactions
    rows: List[Dict[str, Any]] = []

    for art, sent in zip(all_articles, sentiment_outputs):
        sym = art["symbol"]
        hist_df, proxy_used = histories.get(sym, (pd.DataFrame(), None))

        price_reaction = reaction_from_history(
            hist=hist_df,
            published_dt=art.get("published_at"),
            window_hours=cfg.price_window_hours,
            flat_threshold_pct=cfg.flat_threshold_pct,
        )

        if price_reaction:
            before_close = price_reaction["before_close"]
            after_close = price_reaction["after_close"]
            change_pct = price_reaction["change_pct"]
            graph_timestamps = price_reaction["graph_timestamps"]
            graph_closes = price_reaction["graph_closes"]
            price_avail = True
        else:
            before_close = None
            after_close = None
            change_pct = None
            graph_timestamps = []
            graph_closes = []
            price_avail = False

        sentiment_label = sent["label"]
        suggestion = generate_trading_suggestion(
            sentiment=sentiment_label,
            change_pct=change_pct,
            flat_threshold_pct=cfg.flat_threshold_pct,
            price_available=price_avail,
        )
        correct_val = is_correct(
            sentiment=sentiment_label,
            change_pct=change_pct,
            flat_threshold_pct=cfg.flat_threshold_pct,
            price_available=price_avail,
        )

        rows.append({
            "title": art["title"],
            "url": art.get("url", "#"),
            "source": art.get("source", "News"),
            "published_at": art.get("published_at"),
            "company": art["company"],
            "symbol": sym,
            "sentiment": sentiment_label,
            "confidence": sent["confidence"],
            "probabilities": sent["probabilities"],
            "before_close": before_close,
            "after_close": after_close,
            "change_pct": change_pct,
            "suggestion": suggestion,
            "correct": correct_val,
            "proxy": proxy_used,
            "price_available": price_avail,
            "is_demo": False,
            "graph_timestamps": graph_timestamps,
            "graph_closes": graph_closes,
        })

    # Sort rows newest first
    rows.sort(key=lambda r: str(r.get("published_at", "")), reverse=True)

    # 5. Aggregate metrics
    metrics = aggregate_accuracy(rows)
    metrics["articles_analyzed"] = len(rows)
    metrics["positive_count"] = sum(1 for r in rows if r["sentiment"] == "Positive")
    metrics["negative_count"] = sum(1 for r in rows if r["sentiment"] == "Negative")
    metrics["neutral_count"] = sum(1 for r in rows if r["sentiment"] == "Neutral")

    # 6. Companies summary
    companies_summary: Dict[str, Any] = {}
    for comp in cfg.companies:
        comp_rows = [r for r in rows if r["company"] == comp.name]
        pos_n = sum(1 for r in comp_rows if r["sentiment"] == "Positive")
        neg_n = sum(1 for r in comp_rows if r["sentiment"] == "Negative")
        neu_n = sum(1 for r in comp_rows if r["sentiment"] == "Neutral")
        avg_conf = (
            round(sum(r["confidence"] for r in comp_rows) / len(comp_rows), 2)
            if comp_rows
            else 0.0
        )
        valid_moves = [r["change_pct"] for r in comp_rows if r["change_pct"] is not None]
        avg_move = (
            round(sum(valid_moves) / len(valid_moves), 2)
            if valid_moves
            else 0.0
        )
        latest_sugg = comp_rows[0]["suggestion"] if comp_rows else "Hold"
        comp_acc = metrics["by_company"].get(comp.name, {"accuracy_pct": 0.0, "correct": 0, "total": 0})

        companies_summary[comp.name] = {
            "name": comp.name,
            "symbol": comp.symbol,
            "etf": comp.etf,
            "article_count": len(comp_rows),
            "sentiment_counts": {"Positive": pos_n, "Negative": neg_n, "Neutral": neu_n},
            "avg_confidence": avg_conf,
            "avg_move_pct": avg_move,
            "accuracy_pct": comp_acc["accuracy_pct"],
            "correct_predictions": comp_acc["correct"],
            "total_evaluated": comp_acc["total"],
            "latest_suggestion": latest_sugg,
        }

    results_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": data_source,
        "news_provider": cfg.news_provider,
        "model_name": engine.model_description if engine else "FinBERT (fine-tuned)",
        "metrics": metrics,
        "companies": companies_summary,
        "rows": rows,
    }

    _save_results(cfg, results_payload)
    return results_payload


def _save_results(cfg: AppConfig, payload: Dict[str, Any]) -> None:
    """Save results payload to results.json and export to CSV."""
    out_dir = cfg.root_dir / "data" / "processed" / "realtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_file = out_dir / "results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Flatten rows to CSV
    csv_rows = []
    for r in payload.get("rows", []):
        flat = dict(r)
        flat.pop("graph_timestamps", None)
        flat.pop("graph_closes", None)
        flat.pop("probabilities", None)
        csv_rows.append(flat)

    if csv_rows:
        csv_file = out_dir / "results.csv"
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_file, index=False)
