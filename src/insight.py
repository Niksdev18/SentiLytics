"""Insight engine for trading suggestions and direction accuracy scoring.

Pure functions implementing:
- Suggestion Matrix:
    Positive + up   -> "Consider Buy"
    Negative + down -> "Consider Sell"
    Positive + down OR Negative + up -> "Watch - possible delayed reaction"
    Neutral/Mixed OR |change| <= flat_threshold -> "Hold"
    No price data   -> "Hold - no price data"
- Accuracy Matrix:
    positive <-> up (> +flat_threshold)
    negative <-> down (< -flat_threshold)
    neutral  <-> flat (|change| <= flat_threshold)
    Excluded: Mixed and no-price rows
"""

from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def generate_trading_suggestion(
    sentiment: str,
    change_pct: Optional[float],
    flat_threshold_pct: float = 0.5,
    price_available: bool = True,
) -> str:
    """Generate rule-based trading suggestions from sentiment and price change.

    Args:
        sentiment: Sentiment label ("Positive", "Negative", "Neutral", "Mixed").
        change_pct: Stock percentage change.
        flat_threshold_pct: Threshold below which price movement is considered flat.
        price_available: Whether price reaction data was successfully retrieved.

    Returns:
        One of: 'Consider Buy', 'Consider Sell', 'Watch - possible delayed reaction',
        'Hold', 'Hold - no price data'.
    """
    if not price_available or change_pct is None:
        return "Hold - no price data"

    sent_clean = str(sentiment).strip().lower()

    # Movement is within flat bounds or sentiment is neutral/mixed
    if abs(change_pct) <= flat_threshold_pct or sent_clean in ("neutral", "mixed"):
        return "Hold"

    if sent_clean == "positive":
        if change_pct > flat_threshold_pct:
            return "Consider Buy"
        elif change_pct < -flat_threshold_pct:
            return "Watch - possible delayed reaction"

    elif sent_clean == "negative":
        if change_pct < -flat_threshold_pct:
            return "Consider Sell"
        elif change_pct > flat_threshold_pct:
            return "Watch - possible delayed reaction"

    return "Hold"


def is_correct(
    sentiment: str,
    change_pct: Optional[float],
    flat_threshold_pct: float = 0.5,
    price_available: bool = True,
) -> Optional[bool]:
    """Evaluate whether sentiment prediction matched market direction.

    Rules:
    - Positive <-> up (> +threshold)
    - Negative <-> down (< -threshold)
    - Neutral  <-> flat (|change| <= threshold)
    - Excluded from evaluation: missing price data, Mixed sentiment.

    Returns:
        True if correct, False if incorrect, None if excluded.
    """
    if not price_available or change_pct is None:
        return None

    sent_clean = str(sentiment).strip().lower()
    if sent_clean == "mixed":
        return None

    if sent_clean == "positive":
        return bool(change_pct > flat_threshold_pct)
    elif sent_clean == "negative":
        return bool(change_pct < -flat_threshold_pct)
    elif sent_clean == "neutral":
        return bool(abs(change_pct) <= flat_threshold_pct)

    return False


def aggregate_accuracy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate and per-company direction accuracy metrics.

    Args:
        rows: List of article dicts with keys 'company' and 'correct'.

    Returns:
        Dict with overall metrics and by_company breakdown.
    """
    overall_correct = 0
    overall_total = 0
    by_comp: Dict[str, Dict[str, int]] = {}

    for r in rows:
        company = r.get("company", "Unknown")
        if company not in by_comp:
            by_comp[company] = {"correct": 0, "total": 0}

        c_val = r.get("correct")
        if c_val is not None:
            overall_total += 1
            by_comp[company]["total"] += 1
            if c_val is True:
                overall_correct += 1
                by_comp[company]["correct"] += 1

    overall_pct = round((overall_correct / overall_total) * 100.0, 1) if overall_total > 0 else 0.0

    comp_summary: Dict[str, Dict[str, Any]] = {}
    for comp, stats in by_comp.items():
        c_tot = stats["total"]
        c_corr = stats["correct"]
        c_pct = round((c_corr / c_tot) * 100.0, 1) if c_tot > 0 else 0.0
        comp_summary[comp] = {
            "accuracy_pct": c_pct,
            "correct": c_corr,
            "total": c_tot,
        }

    return {
        "accuracy_pct": overall_pct,
        "accuracy": overall_pct,
        "correct": overall_correct,
        "correct_predictions": overall_correct,
        "total": overall_total,
        "total_predictions": overall_total,
        "mismatches": overall_total - overall_correct,
        "by_company": comp_summary,
    }
