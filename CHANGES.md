# Senti-Lytics: Changes and Verification Log

This document details the 14 verified problems identified in the original codebase, their root causes, the refactoring solutions implemented across Phases 0–4, and the verification methods used to prove each fix.

---

## Summary of Fixes

| # | Issue Area | Original Buggy State | Refactored Solution | Verification |
|---|------------|----------------------|---------------------|--------------|
| **1** | Startup & Fallbacks | `app.py` crashed on startup if `config.yaml` or local fine-tuned weights were missing; `local_files_only=True` without graceful fallback. | Layered configuration in `src/config.py` (defaults < `config.yaml` < `.env`). `src/sentiment.py` lazy-loads models, provides HuggingFace Hub fallback (`ProsusAI/finbert`) or DEMO mode, reporting status cleanly. | `tests/test_phase2.py`, `tests/test_phase3.py` (no-model & no-key test paths pass with 200/503 responses). |
| **2** | Dependencies | `requirements.txt` missed 10+ runtime packages (`apscheduler`, `vaderSentiment`, `xgboost`, `imbalanced-learn`, `joblib`, `feedparser`, `streamlit`, `plotly`, `pillow`, `python-dotenv`). | `requirements.txt` sanitized with exact pinned versions required for the core app; paper benchmark packages isolated in `requirements-benchmark.txt`. | Clean virtual environment install succeeds; all 49 pytest tests pass. |
| **3** | Template / Backend Contract Mismatch | `dashboard.html` expected fields never produced (`before_close`, `after_close`, `change_24h`, `graph_timestamps`, `graph_closes`, counts). `metrics` was always `None`. | `src/pipeline.py` emits complete per-article rows (`before_close`, `after_close`, `change_pct`, `graph_timestamps`, `graph_closes`, `correct`, `proxy`, `price_available`) and summary `metrics` dict. Template updated to use `change_pct`. | `test_pipeline_run` verifies schema; dashboard renders KPI strip and price cards without errors. |
| **4** | Price Window Alignment | "Before/after" price simply took the last two daily closing prices, completely ignoring when an article was actually published. | `src/prices.py` implements pure function `reaction_from_history(hist, published_dt, window_hours=4)` extracting actual price movement around publication time with timezone and weekend handling. | `tests/test_phase2.py` tests price reaction across hourly/daily intervals, weekend gaps, and market closes. |
| **5** | Train / Serve Text Mismatch | `preprocess.py` aggressively stripped NLTK stopwords (removing "not", "no", "up", "down", "below") and lowercased; FinBERT was fine-tuned on stripped text while `app.py` served raw headlines. | Separated into two clean functions in `src/preprocessing/preprocess.py`: `light_clean` (preserves casing, punctuation, and negation for BERT) and `classic_clean` (preserves negation and market-direction tokens for TF-IDF). | `test_light_clean_preserves_negation_and_direction` in `tests/test_phase1.py` proves "not", "down", "up" are retained. |
| **6** | Side-Effects at Import | `preprocess.py` ran disk I/O and NLTK downloads on import; attempted to read `data/raw/financial_phrasebank_clean.csv` which was never generated. | Created `src/preprocessing/prepare_dataset.py` to parse raw `data/Sentences_AllAgree.txt` (latin-1, `rsplit` on `@`). All scripts guard execution under `if __name__ == '__main__':`. | Dataset parser produces 2,264 clean rows (1,391 neutral, 570 positive, 303 negative) verified in `tests/test_phase1.py`. |
| **7** | Label Mapping & Model Metadata | Hardcoded `{0:Negative, 1:Neutral, 2:Positive}` in `app.py`. `transformer.py` did not persist `id2label` into model config or export `metrics.json`. | Unified canonical mapping (`0: negative`, `1: neutral`, `2: positive`). `src/models/transformer.py` writes `id2label`/`label2id` into HuggingFace config and saves comprehensive `metrics.json`. | `test_transformer_config_metadata` in `tests/test_phase1.py` verifies round-trip persistence. |
| **8** | Fabricated Metrics in Templates | Templates had hardcoded, conflicting accuracy figures ("90.03%" vs "92.14%"), hand-typed 7-model comparison table, and phantom claims ("Deployed on Render"). | Removed all hardcoded metric numbers from `index.html`, `dashboard.html`, and `about.html`. All metrics read dynamically from `results.json`, `models/model_metrics.csv`, or live pipeline. If unavailable, UI displays "Not available - run training". | Grep check confirms zero occurrences of `90.03`, `92.14`, or `86.71` across all templates. |
| **9** | Redundant Inference & Insecure API Calls | `app.py` called sentiment inference twice per headline, passed API key in URL parameters, used unbounded `requests.get` without timeout, and broad queries like `q=Apple`. | Sentiment is scored once in batch with `torch.no_grad()`. `src/news.py` uses `X-Api-Key` header, configurable timeout (10s), 5xx retry logic, title deduplication, and targeted boolean queries (`"Apple" AND (stock OR shares OR earnings)`). | Network calls contain timeouts; unit tests verify header-based authentication and query construction. |
| **10** | NewsAPI Free Tier Quota Exhaustion | Monolithic scheduler queried 10 companies every 2 hours = 120 calls/day, exceeding the 100/day free limit. | `src/config.py` enforces quota budgeting: warns at startup if `companies * (24 / refresh_hours) > 100`. Default tracking set to 6 core companies with caching in `data/raw/newsapi/`. | Config startup validator runs quota calculation on initialization. |
| **11** | Ad-Hoc Probability Distortions | `predict_sentiment` artificially forced neutral probabilities < 0.70 into Positive/Negative and blended unvalidated VADER overrides. | Removed heuristic biases from default inference path. Clean FinBERT softmax probabilities are used. Legacy heuristics isolated as optional config flags defaulting to `False`. | `src/sentiment.py` outputs calibrated probabilities; `tests/test_phase2.py` validates unbiased argmax outputs. |
| **12** | Flawed Heuristics & Inverted Labels | `predictor.py` used substring matches (e.g., "fine" matched "define"); negation inverted any keyword if "not" appeared anywhere; `rss_market_sentiment.py` used `{2:-1, 1:1, 0:0}`. | Retired buggy heuristic files to `legacy/`. Label space standardized: 0=negative, 1=neutral, 2=positive across all models, pipeline stages, and UI visualizers. | `legacy/README.md` documents retired files; `tests/test_smoke.py` validates canonical label consistency. |
| **13** | Incomplete Signal Logic | Only BUY/SELL/HOLD were returned; the "Watch (delayed reaction)" thesis and actual directional accuracy metric were missing. | `src/insight.py` implements complete matrix: Positive+up → "Consider Buy", Negative+down → "Consider Sell", Positive+down or Negative+up → "Watch - possible delayed reaction", Neutral/flat → "Hold". Implemented `is_correct()` and `aggregate_accuracy()`. | `tests/test_phase2.py` unit tests all 6 branches of the suggestion matrix and directional correctness checks. |
| **14** | Unsound Streamlit Dashboard | Streamlit app had hardcoded `admin/admin` credentials, hardcoded 85% accuracy, and passed lists to functions expecting strings. | Retired to `legacy/streamlit_app.py`. Built a unified, modular Flask application factory (`src/api/app.py`) with background scheduler, REST endpoints, and responsive dark-theme dashboard. | `tests/test_phase3.py` validates Flask app factory, REST API endpoints, `/health`, `/refresh`, and error handlers. |

---

## Detailed Issue Walkthrough

### 1. Configuration & Graceful Fallback
- **Problem**: Original `app.py` failed immediately when executed in a fresh checkout. It looked for `config.yaml` (which didn't exist) and attempted to load a local model directory that was not present.
- **Solution**: Implemented hierarchical configuration in `src/config.py` using Pydantic/dataclasses. The priority order is: built-in safe defaults -> `config.yaml` (if present) -> environment variables (`.env`). If no fine-tuned local model is present, the app can either fall back to HuggingFace Hub's `ProsusAI/finbert` (if internet access is enabled) or run in self-contained DEMO mode (`DEMO_MODE=True`), showing a visible banner.

### 2. Missing Dependencies
- **Problem**: Running `pip install -r requirements.txt` left the environment unable to import `apscheduler`, `xgboost`, `imbalanced-learn`, and `python-dotenv`.
- **Solution**: Audited all imports across the codebase. Kept `requirements.txt` lean for production execution while maintaining test dependencies in `requirements.txt` and separating heavy paper benchmarking libraries (`matplotlib`, `statsmodels`) into `requirements-benchmark.txt`.

### 3. Data Contract Alignment
- **Problem**: The frontend UI template `dashboard.html` contained Jinja2 expressions referencing variables like `article.before_close`, `article.after_close`, `article.change_24h`, and `metrics.positive_count`. The backend never supplied these, rendering empty cards and broken price charts.
- **Solution**: Structured the pipeline output in `src/pipeline.py` into a strictly validated dictionary containing both aggregate statistics and individual headline analysis dictionaries with price history coordinates for Chart.js rendering.

### 4. Realistic Price Reaction Windows
- **Problem**: The original market module pulled daily closing quotes for "yesterday" and "today", claiming this represented the market impact of an article published at 2:15 PM today.
- **Solution**: `src/prices.py` fetches intraday/hourly bars via `yfinance` around `article.published_at` within a configurable window (`price_window_hours=4`). If publication happens outside market hours (nights/weekends), the window aligns to the next market open. If stock ticker data is unavailable, the system transparently falls back to the relevant sector ETF (e.g., `XLK` for tech) and flags the record with `proxy="XLK"`.

### 5. Training vs Serving Text Preprocessing
- **Problem**: Stopwords such as "not", "down", "up", and "below" carry vital financial sentiment signals. Stripping them before training BERT severely handicapped the transformer, while the inference server passed raw text.
- **Solution**: Introduced `light_clean()` in `src/preprocessing/preprocess.py` for BERT/transformer pipelines (strips HTML, normalizes spaces, preserves case and grammar tokens). For classical bag-of-words/TF-IDF models, `classic_clean()` lowercases and removes generic stopwords while explicitly preserving negation and financial direction tokens (`no`, `not`, `nor`, `up`, `down`, `over`, `under`, `above`, `below`, `off`).

### 6. Import Side-Effects & Dataset Generation
- **Problem**: Importing any helper from `preprocess.py` triggered immediate dataset loading and attempted to read from an uncommitted path `data/raw/financial_phrasebank_clean.csv`.
- **Solution**: Created `src/preprocessing/prepare_dataset.py` which parses the raw Latin-1 encoded `data/Sentences_AllAgree.txt` using reverse-split on the `@` delimiter. Removed all top-level executable code from library modules, encapsulating execution in testable CLI entrypoints.

### 7. Explicit Label Encodings & Metrics Tracking
- **Problem**: Integer mappings were assumed inconsistently across scripts, and fine-tuned models did not write `id2label` to their HuggingFace `config.json`.
- **Solution**: Standardized `CLASS_ORDER = ["negative", "neutral", "positive"]` across all modules. `src/models/transformer.py` stores label mappings in the model configuration, exports `metrics.json` (precision, recall, macro-F1, weighted-F1), and appends training history to `models/model_metrics.csv`.

### 8. Removal of Fabricated Metrics
- **Problem**: The UI displayed hardcoded metrics claiming 90.03% or 92.14% accuracy, along with an unverified 7-model comparison table and claims of deployment on Render.
- **Solution**: Eliminated all static metric numbers. The dashboard now dynamically reads live metrics from `results.json` and historical benchmark numbers from `models/model_metrics.csv`. When no model has been trained, the interface displays an informative "Not available - run training" state.

### 9. Secure News Ingestion & Single-Pass Inference
- **Problem**: NewsAPI was called with credentials embedded in URL query parameters, without timeouts, and using overly broad search strings that returned irrelevant entertainment or generic news.
- **Solution**: `src/news.py` places API keys in the `X-Api-Key` HTTP header, applies a 10-second timeout, handles HTTP 429/5xx status codes with retry backoff, deduplicates syndicated headlines, and constructs targeted queries (e.g., `"Microsoft" AND (stock OR shares OR earnings)`). Headlines are passed through FinBERT in a single batch.

### 10. NewsAPI Free Tier Rate Limiting
- **Problem**: Polling 10 companies every 2 hours resulted in 120 requests per day, exceeding the NewsAPI free tier limit of 100 calls per day.
- **Solution**: Configured tracking for 6 primary companies by default. Added an automated startup validation in `src/config.py` that calculates `companies * (24 / refresh_hours)` and emits a warning if the planned schedule exceeds 100 requests/day. Added raw JSON response caching in `data/raw/newsapi/`.

### 11. Objective Confidence & Sentiment Scoring
- **Problem**: Arbitrary thresholds forced any prediction with neutral probability < 0.70 into a directional call, distorting model calibration.
- **Solution**: Standard inference outputs true softmax probabilities from FinBERT. The raw argmax determines the predicted label, providing an honest baseline. Optional heuristic overrides remain available via configuration flags for research experimentation.

### 12. Heuristic Isolation & Codebase Cleanup
- **Problem**: `predictor.py` contained substring matching bugs where words like "define" triggered the positive keyword "fine".
- **Solution**: Deprecated and moved broken heuristic scripts into `legacy/` accompanied by `legacy/README.md`.

### 13. Directional Accuracy & "Watch" Signal
- **Problem**: The project claimed to track market reaction accuracy and suggest "Watch" on delayed reactions, but the code only implemented basic Buy/Sell/Hold.
- **Solution**: Implemented `src/insight.py` with pure functions:
  - Positive sentiment + stock price up $\rightarrow$ `Consider Buy`
  - Negative sentiment + stock price down $\rightarrow$ `Consider Sell`
  - Positive sentiment + stock price down $\rightarrow$ `Watch - possible delayed reaction`
  - Negative sentiment + stock price up $\rightarrow$ `Watch - possible delayed reaction`
  - Neutral sentiment or price change $< \pm 0.5\%$ $\rightarrow$ `Hold`
  - Missing price data $\rightarrow$ `Hold - no price data`
  - Evaluates `is_correct` by checking alignment between sentiment polarity and price direction.

### 14. Architecture Modernization & Testing
- **Problem**: Streamlit app contained hardcoded credentials and lacked automated tests.
- **Solution**: Standardized on a clean Flask application factory in `src/api/app.py`. Added comprehensive unit and integration test suites in `tests/` covering dataset preparation, preprocessing, backend pipeline, price calculation, suggestion logic, and Flask REST endpoints.

