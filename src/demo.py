"""Deterministic synthetic demo data generator.

Features:
- Deterministic seeding (random.Random(42)) for 100% reproducible demo states
- Generates realistic articles for all configured companies
- Every headline marked with [DEMO] and tagged is_demo=True
- Balanced, realistic distribution of correct vs incorrect signals, Buy/Sell/Hold/Watch,
  and ETF proxy tags.
"""

from datetime import datetime, timezone, timedelta
import random
from typing import Dict, Any, List, Optional

from src.config import AppConfig, DEFAULT_COMPANIES
from src.insight import generate_trading_suggestion, is_correct, aggregate_accuracy

DEMO_HEADLINE_TEMPLATES = [
    ("{name} reports record quarterly earnings, beating street expectations", "Positive", 2.45),
    ("{name} unveils breakthrough generative AI architecture in product line", "Positive", 1.80),
    ("{name} faces antitrust regulatory probe and potential fines in EU", "Negative", -2.15),
    ("{name} trims revenue guidance amid softening enterprise hardware demand", "Negative", -3.10),
    ("{name} holds annual shareholder meeting; reiterates full-year fiscal targets", "Neutral", 0.15),
    ("{name} announces partnership with regional telecom providers", "Positive", -1.20),  # Delayed reaction / mismatch
    ("{name} supplier halts production temporarily following factory audit", "Negative", 1.05),   # Delayed reaction / mismatch
]


def generate_demo_results(config: Optional[AppConfig] = None) -> Dict[str, Any]:
    """Generate deterministic synthetic demo results for offline presentations.

    Args:
        config: AppConfig instance.

    Returns:
        Dict adhering to results.json schema with 'rows', 'metrics', 'companies'.
    """
    cfg = config or AppConfig()
    if not cfg.companies:
        cfg.companies = list(DEFAULT_COMPANIES)
    rng = random.Random(42)

    rows: List[Dict[str, Any]] = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=6)

    # Base price pegs for realistic pricing
    base_prices = {
        "AAPL": 225.50,
        "TSLA": 254.20,
        "AMZN": 186.40,
        "GOOGL": 168.90,
        "MSFT": 428.10,
        "META": 585.30,
        "NVDA": 132.80,
        "RELIANCE.NS": 2840.00,
        "HDFCBANK.NS": 1670.00,
        "TATAMOTORS.NS": 960.00,
    }

    for comp_idx, comp in enumerate(cfg.companies):
        base_price = base_prices.get(comp.symbol, 150.0)
        # Generate 2 to 3 articles per company
        templates = rng.sample(DEMO_HEADLINE_TEMPLATES, 3)

        for art_idx, (tmpl, sentiment, price_shift_pct) in enumerate(templates):
            headline = tmpl.format(name=comp.name)
            pub_time = base_time + timedelta(minutes=comp_idx * 25 + art_idx * 15)

            # Add minor deterministic jitter
            jitter = (rng.random() - 0.5) * 0.4
            final_change_pct = round(price_shift_pct + jitter, 2)

            before_close = round(base_price + (rng.random() - 0.5) * 10, 2)
            after_close = round(before_close * (1.0 + final_change_pct / 100.0), 2)

            # Confidence probabilities
            conf = round(rng.uniform(0.72, 0.94), 2)
            rem = round((1.0 - conf) / 2.0, 4)
            if sentiment == "Positive":
                probs = {"Positive": conf, "Neutral": rem, "Negative": rem}
            elif sentiment == "Negative":
                probs = {"Negative": conf, "Neutral": rem, "Positive": rem}
            else:
                probs = {"Neutral": conf, "Positive": rem, "Negative": rem}

            suggestion = generate_trading_suggestion(
                sentiment=sentiment,
                change_pct=final_change_pct,
                flat_threshold_pct=cfg.flat_threshold_pct,
                price_available=True,
            )
            correct_val = is_correct(
                sentiment=sentiment,
                change_pct=final_change_pct,
                flat_threshold_pct=cfg.flat_threshold_pct,
                price_available=True,
            )

            # Sparkline dummy curve
            graph_closes = [
                round(before_close * (1 + (i - 3) * (final_change_pct / 600.0)), 2)
                for i in range(7)
            ]
            graph_timestamps = [
                (pub_time + timedelta(hours=i - 3)).strftime("%Y-%m-%d %H:%M")
                for i in range(7)
            ]

            proxy_tag = comp.etf if comp_idx % 4 == 0 else None

            rows.append({
                "title": f"[DEMO] {headline}",
                "url": "#",
                "source": "Demo Financial Wire",
                "published_at": pub_time.isoformat(),
                "company": comp.name,
                "symbol": comp.symbol,
                "sentiment": sentiment,
                "confidence": conf,
                "probabilities": probs,
                "before_close": before_close,
                "after_close": after_close,
                "change_pct": final_change_pct,
                "suggestion": suggestion,
                "correct": correct_val,
                "proxy": proxy_tag,
                "price_available": True,
                "is_demo": True,
                "graph_timestamps": graph_timestamps,
                "graph_closes": graph_closes,
            })

    # Sort descending by published_at
    rows.sort(key=lambda r: r["published_at"], reverse=True)

    metrics = aggregate_accuracy(rows)

    companies_summary = {}
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
        avg_move = (
            round(sum(r["change_pct"] for r in comp_rows) / len(comp_rows), 2)
            if comp_rows
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

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "demo",
        "model_name": "FinBERT (synthetic demo)",
        "metrics": metrics,
        "companies": companies_summary,
        "rows": rows,
    }
