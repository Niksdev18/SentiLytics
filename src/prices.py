"""Stock price retrieval and market reaction analysis.

Pure functions for computing price reactions around news publication times,
handling timezone conversions, weekend/market closures, intraday vs daily bars,
and ETF proxy fallbacks.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple, List
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _to_utc_timestamp(dt: Any) -> pd.Timestamp:
    """Normalize a datetime, string, or pd.Timestamp into tz-aware UTC pd.Timestamp."""
    if isinstance(dt, str):
        # parse string with dateutil/pandas
        ts = pd.to_datetime(dt)
    elif isinstance(dt, datetime):
        ts = pd.Timestamp(dt)
    elif isinstance(dt, pd.Timestamp):
        ts = dt
    else:
        raise ValueError(f"Unsupported datetime type: {type(dt)}")

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def reaction_from_history(
    hist: pd.DataFrame,
    published_dt: Any,
    window_hours: int = 4,
    flat_threshold_pct: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Pure function: calculate stock price reaction to news published at published_dt.

    Args:
        hist: DataFrame containing historical prices with DatetimeIndex and 'Close' column.
        published_dt: News publication timestamp (ISO string, datetime, or Timestamp).
        window_hours: Hours after publication to measure reaction.
        flat_threshold_pct: Percentage threshold below which moves are deemed flat.

    Returns:
        Dict with keys: before_close, after_close, change_pct, graph_timestamps,
        graph_closes, price_available. Returns None if hist has insufficient data.
    """
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    # Work on a copy with tz-aware UTC DatetimeIndex
    df = hist.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()

    try:
        pub_utc = _to_utc_timestamp(published_dt)
    except Exception as e:
        logger.warning("Failed to parse published_dt %s: %s", published_dt, e)
        return None

    target_utc = pub_utc + timedelta(hours=window_hours)

    # 1. Determine before_close: latest bar on or before published_dt
    bars_before = df[df.index <= pub_utc]
    if not bars_before.empty:
        before_bar = bars_before.iloc[-1]
        before_idx = bars_before.index[-1]
    else:
        # If published before earliest recorded bar, take the first available bar
        before_bar = df.iloc[0]
        before_idx = df.index[0]

    before_close = float(before_bar["Close"])

    # 2. Determine after_close: earliest bar at or after target_utc
    bars_after = df[df.index >= target_utc]
    if not bars_after.empty:
        after_bar = bars_after.iloc[0]
        after_idx = bars_after.index[0]
    else:
        # Fall back to the latest bar available after before_idx
        subsequent = df[df.index > before_idx]
        if not subsequent.empty:
            after_bar = subsequent.iloc[-1]
            after_idx = subsequent.index[-1]
        else:
            # Only current bar available (e.g. published right now)
            after_bar = before_bar
            after_idx = before_idx

    after_close = float(after_bar["Close"])

    # 3. Calculate % change
    if before_close > 0:
        change_pct = round(((after_close - before_close) / before_close) * 100.0, 2)
    else:
        change_pct = 0.0

    # 4. Generate sparkline / mini graph points (up to 15 bars around publication)
    start_window = before_idx - timedelta(days=2)
    end_window = after_idx + timedelta(days=2)
    graph_slice = df[(df.index >= start_window) & (df.index <= end_window)]
    if graph_slice.empty or len(graph_slice) < 2:
        graph_slice = df.tail(15)

    graph_timestamps: List[str] = [ts.strftime("%Y-%m-%d %H:%M") for ts in graph_slice.index]
    graph_closes: List[float] = [round(float(c), 2) for c in graph_slice["Close"]]

    return {
        "before_close": round(before_close, 2),
        "after_close": round(after_close, 2),
        "change_pct": change_pct,
        "graph_timestamps": graph_timestamps,
        "graph_closes": graph_closes,
        "price_available": True,
    }


def fetch_symbol_history(
    symbol: str,
    proxy_etf: Optional[str] = None,
    period: str = "5d",
    interval: str = "1h",
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch history for symbol; fall back to proxy ETF if ticker data is empty.

    Args:
        symbol: Stock ticker (e.g. 'AAPL', 'RELIANCE.NS').
        proxy_etf: Fallback sector ETF (e.g. 'XLK', 'INDA').
        period: YFinance history period.
        interval: Bar interval (e.g. '1h' for intraday, '1d' for daily).

    Returns:
        Tuple of (DataFrame, proxy_used_or_None).
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist is not None and not hist.empty and len(hist) >= 2:
            return hist, None
    except Exception as e:
        logger.warning("Error fetching ticker %s: %s", symbol, e)

    # Fallback to proxy ETF
    if proxy_etf:
        try:
            logger.info("Attempting ETF fallback %s for %s", proxy_etf, symbol)
            ticker_etf = yf.Ticker(proxy_etf)
            etf_hist = ticker_etf.history(period=period, interval=interval)
            if etf_hist is not None and not etf_hist.empty and len(etf_hist) >= 2:
                return etf_hist, proxy_etf
        except Exception as e:
            logger.warning("Error fetching proxy ETF %s: %s", proxy_etf, e)

    return pd.DataFrame(), None
