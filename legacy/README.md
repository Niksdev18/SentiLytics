# Legacy Files

These files were superseded during the Phase 0–5 refactor and moved here for reference.

| File | Original Location | Reason |
|------|------------------|--------|
| `streamlit_app.py` | `dashboard/app.py` | Replaced by Flask-only dashboard; hardcoded admin/admin login, hardcoded accuracy, type bug |
| `fetch_news_job.py` | `src/api/fetch_news_job.py` | Duplicate pipeline; functionality merged into `src/pipeline.py` |
| `predict_realtime.py` | `src/inference/predict_realtime.py` | Duplicate pipeline; replaced by `src/sentiment.py` + `src/pipeline.py` |
| `news_fetcher_rss.py` | `src/utils/news_fetcher.py` | Google RSS fetcher; replaced by `src/news.py` (NewsAPI with header auth) |
| `market_sentiment.py` | `src/utils/market_sentiment.py` | Hardcoded TSLA; inverted label map; replaced by `src/prices.py` |
| `predictor_enhanced.py` | `src/models/predictor.py` | Unused heuristics (VADER blend, keyword override); substring matching bugs |
| `preprocess_realtime.py` | `src/preprocessing/preprocess_realtime.py` | Triggered file I/O at import via `preprocess.py` |
