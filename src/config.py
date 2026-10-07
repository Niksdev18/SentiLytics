"""Central configuration module.

Hierarchical loading: hardcoded defaults < config.yaml < environment variables.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import os
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CompanyConfig:
    name: str
    symbol: str
    query: str
    etf: str


DEFAULT_COMPANIES = [
    CompanyConfig("Apple", "AAPL", '"Apple" AND (stock OR shares OR earnings OR AAPL)', "XLK"),
    CompanyConfig("Tesla", "TSLA", '"Tesla" AND (stock OR shares OR earnings OR TSLA)', "CARZ"),
    CompanyConfig("Amazon", "AMZN", '"Amazon" AND (stock OR shares OR earnings OR AMZN)', "XLY"),
    CompanyConfig("Google", "GOOGL", '"Alphabet" OR "Google" AND (stock OR shares OR earnings OR GOOGL)', "XLC"),
    CompanyConfig("Microsoft", "MSFT", '"Microsoft" AND (stock OR shares OR earnings OR MSFT)', "XLK"),
    CompanyConfig("Meta", "META", '"Meta Platforms" AND (stock OR shares OR earnings OR META)', "XLC"),
    CompanyConfig("NVIDIA", "NVDA", '"NVIDIA" AND (stock OR shares OR earnings OR NVDA)', "SMH"),
    CompanyConfig("Reliance", "RELIANCE.NS", '"Reliance Industries" AND (stock OR shares OR earnings)', "INDA"),
    CompanyConfig("HDFC Bank", "HDFCBANK.NS", '"HDFC Bank" AND (stock OR shares OR earnings)', "INDA"),
    CompanyConfig("Tata Motors", "TATAMOTORS.NS", '"Tata Motors" AND (stock OR shares OR earnings)', "INDA"),
]


@dataclass
class AppConfig:
    # API Keys
    newsapi_key: Optional[str] = None

    # Model Settings
    model_path: str = "models/transformer_model/best_model"
    allow_hub_fallback: bool = True

    # Pipeline Settings
    refresh_hours: int = 2
    articles_per_company: int = 5
    flat_threshold_pct: float = 0.5
    price_window_hours: int = 4

    # Companies
    companies: List[CompanyConfig] = field(default_factory=lambda: list(DEFAULT_COMPANIES))

    # Heuristics (disabled by default for scientific honesty)
    neutral_floor_enabled: bool = False
    neutral_floor_threshold: float = 0.70
    vader_assist_enabled: bool = False

    # Demo mode
    demo_mode: bool = False

    # Directories
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration with hierarchy: defaults < config.yaml < env vars."""
    root = Path(__file__).resolve().parent.parent

    # Try .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass

    cfg = AppConfig()

    # Load YAML if present
    if config_path is None:
        yaml_candidates = [root / "config.yaml", root / "config_template.yaml"]
        for cand in yaml_candidates:
            if cand.exists():
                config_path = cand
                break

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # API Keys
            if "api_keys" in data and isinstance(data["api_keys"], dict):
                cfg.newsapi_key = data["api_keys"].get("newsapi")

            # Model
            if "model" in data and isinstance(data["model"], dict):
                cfg.model_path = data["model"].get("path", cfg.model_path)
                cfg.allow_hub_fallback = data["model"].get("allow_hub_fallback", cfg.allow_hub_fallback)

            # Pipeline
            if "pipeline" in data and isinstance(data["pipeline"], dict):
                cfg.refresh_hours = int(data["pipeline"].get("refresh_hours", cfg.refresh_hours))
                cfg.articles_per_company = int(data["pipeline"].get("articles_per_company", cfg.articles_per_company))
                cfg.flat_threshold_pct = float(data["pipeline"].get("flat_threshold_pct", cfg.flat_threshold_pct))
                cfg.price_window_hours = int(data["pipeline"].get("price_window_hours", cfg.price_window_hours))

            # Companies
            if "companies" in data and isinstance(data["companies"], list):
                parsed_companies = []
                for c in data["companies"]:
                    parsed_companies.append(
                        CompanyConfig(
                            name=c.get("name", ""),
                            symbol=c.get("symbol", ""),
                            query=c.get("query", ""),
                            etf=c.get("etf", "SPY"),
                        )
                    )
                if parsed_companies:
                    cfg.companies = parsed_companies

            # Heuristics
            if "heuristics" in data and isinstance(data["heuristics"], dict):
                cfg.neutral_floor_enabled = bool(data["heuristics"].get("neutral_floor_enabled", cfg.neutral_floor_enabled))
                cfg.neutral_floor_threshold = float(data["heuristics"].get("neutral_floor_threshold", cfg.neutral_floor_threshold))
                cfg.vader_assist_enabled = bool(data["heuristics"].get("vader_assist_enabled", cfg.vader_assist_enabled))

            # Demo
            if "demo" in data and isinstance(data["demo"], dict):
                cfg.demo_mode = bool(data["demo"].get("enabled", cfg.demo_mode))

        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)

    # Environment variables override YAML and defaults
    env_newsapi = os.environ.get("NEWSAPI_KEY")
    if env_newsapi:
        cfg.newsapi_key = env_newsapi

    env_model_path = os.environ.get("MODEL_PATH")
    if env_model_path:
        cfg.model_path = env_model_path

    env_demo = os.environ.get("DEMO_MODE")
    if env_demo is not None:
        cfg.demo_mode = env_demo.strip().lower() in ("1", "true", "yes")

    # Rate limit check: NewsAPI free tier allows ~100 calls/day
    daily_calls = len(cfg.companies) * (24 // max(1, cfg.refresh_hours))
    if daily_calls > 100:
        logger.warning(
            f"Quota alert: Tracking {len(cfg.companies)} companies every {cfg.refresh_hours}h "
            f"requires {daily_calls} calls/day, which exceeds the NewsAPI free tier limit (100 calls/day)."
        )

    return cfg
