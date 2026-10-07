<USER_REQUEST>
# ROLE
You are a senior Python/ML + Flask engineer and UI polisher. Enhance the existing
"Senti-Lytics" repo in this workspace into a reliable, demo-ready Financial News
Sentiment Dashboard. I present in ~2 days, so a working, honest, good-looking app
beats a big one. Work in the phases below. After each phase, RUN the commands,
paste real output, and STOP until I reply "next". Never say "should work".

# WHAT THE PROJECT IS
Fetch financial headlines per company (NewsAPI) -> classify sentiment with FinBERT
(fine-tuned in src/models/transformer.py) -> fetch stock prices around each
article's publish time (yfinance) -> compute % change -> produce a Buy/Sell/Watch/
Hold suggestion + a "was sentiment direction right?" accuracy score -> show it on
a Flask dashboard that auto-refreshes every 2 hours (APScheduler).

# REPO MAP (existing)
- src/api/app.py                 Flask app, model loading, news fetch, scheduler (monolith)
- src/api/templates/             index.html, dashboard.html, about.html, dashboard.css (dark UI, Bootstrap 5 + Chart.js CDN)
- src/models/baseline.py         TF-IDF + LR/RF/XGB/SVM/MNB + stacking + voting -> models/model_metrics.csv
- src/models/transformer.py      fine-tunes ProsusAI/finbert -> models/transformer_model/best_model
- src/models/predictor.py        unused "enhanced" heuristics (VADER blend, keyword override)
- src/preprocessing/             preprocess.py, preprocess_realtime.py, split_data.py
- src/utils/                     news_fetcher.py (Google RSS), market_sentiment.py (hardcoded TSLA)
- src/inference/predict_realtime.py, src/api/fetch_news_job.py   duplicate older pipelines
- dashboard/app.py               Streamlit app (hardcoded admin/admin login, hardcoded "85%")
- data/Sentences_AllAgree.txt    Financial PhraseBank, format "sentence@label", latin-1 encoded
- junk: "Order it.py3" (unrelated, syntax errors), a.ipynb (empty), dashboard/components/*.py (empty),
  duplicate project/ and senti-lytics/ folders, ab.xml (draw.io flowchart)

# VERIFIED PROBLEMS (found by reading the code; confirm each, then fix)
1. app.py crashes on start: needs config.yaml (README says rename config_template.yaml, which does not exist)
   and a local model with local_files_only=True. No graceful fallback.
2. requirements.txt is missing imports used in the code: apscheduler, vaderSentiment, xgboost,
   imbalanced-learn, joblib, feedparser, streamlit, plotly, pillow, python-dotenv.
3. dashboard.html expects fields app.py never produces: before_close, after_close, change_24h,
   graph_timestamps, graph_closes, articles_analyzed, positive_count, negative_count, neutral_count.
   `metrics` is always None, so the accuracy strip and price charts never render.
4. "Before/after" price is just the last two daily closes, NOT the price around each article's publish time.
5. TRAIN/SERVE MISMATCH: preprocess.py strips NLTK stopwords (removing "not", "no", "up", "down",
   "over", "under", "below") and lowercases; transformer.py trains BERT on that cleaned text, but app.py
   feeds BERT raw headlines. BERT should train on lightly cleaned (not stopword-stripped) text.
   Classical models may keep aggressive cleaning but must keep negation/direction words.
6. preprocess.py runs file I/O and nltk downloads at import time, so preprocess_realtime.py importing
   clean_text triggers it. It reads data/raw/financial_phrasebank_clean.csv, which nothing creates
   (the raw file is data/Sentences_AllAgree.txt).
7. Label mapping is hardcoded {0:Negative,1:Neutral,2:Positive} in app.py; transformer.py never stores
   id2label in the saved model config. Also writes no metrics.json, but app.py reads MODEL_PATH/metrics.json,
   so MODEL_METRICS is always {}.
8. Hardcoded, mutually inconsistent accuracy claims in templates: "90.03" (dashboard sidebar, about.html)
   vs "92.14" (index.html, about.html, dashboard.html), a hand-typed 7-model comparison table in index.html,
   "Outperforms every baseline by ~4%", and "Deployed on Render / model on HuggingFace Hub" in about.html.
   about.html also advertises POST /api/predict, which does not exist.
9. app.py runs sentiment twice per article (aggregate + per-article), builds the NewsAPI URL with the key
   in the query string, uses requests.get with no timeout, bare `except:`, and `debug=True`.
   Query `q=Apple` / `Amazon` / `Meta` / `Reliance` returns irrelevant results.
10. Quota: 10 companies x every 2h = 120 NewsAPI calls/day, above the free tier's 100/day.
11. Ad-hoc heuristics in predict_sentiment (force Pos/Neg when neutral prob < 0.70; VADER override)
    inflate directional calls and feed the Buy/Sell signal unvalidated.
12. predictor.py keyword matching uses substring tests, so "fine" matches "define"/"refined"; the negation
    rule flips any positive keyword if "not"/"no" appears anywhere. rss_market_sentiment.py maps
    {2:-1, 1:1, 0:0}, contradicting the app's 0=neg,1=neu,2=pos mapping.
13. The suggestion logic only returns BUY/SELL/HOLD; the "Watch (possible delayed reaction)" case and
    the accuracy score described in the project report do not exist in code.
14. Streamlit dashboard/app.py: hardcoded admin/admin login, hardcoded accuracy, and it passes a list to a
    function that expects one string.

# HARD RULES
- HONESTY: never hardcode or invent accuracy/F1/comparison numbers. Every metric shown in the UI must be read
  from files produced by the training/eval scripts (models/model_metrics.csv, metrics.json) or computed live.
  If a metric file is missing, show "Not available - run training", not a made-up number.
- Describe the model truthfully: "FinBERT (fine-tuned)" only if a fine-tuned model + metrics.json exist;
  otherwise "FinBERT (pretrained)".
- DEMO MODE: bundle a clearly labeled synthetic dataset ("DEMO DATA" banner on every page that shows it) so
  the dashboard works offline for the presentation. Demo numbers must never look like real results.
- Do not delete my code: move superseded files into /legacy with a one-line note in legacy/README.md.
  Delete only true junk (Order it.py3, empty a.ipynb, empty components, duplicate project/ and senti-lytics/
  folders; keep one copy of the sample RSS json under data/sample/; keep ab.xml as docs/flowchart.drawio).
- No secrets in code, URLs, or logs. Send the NewsAPI key in the X-Api-Key header. Use .env + config.yaml;
  provide config_template.yaml and .env.example. Add timeouts to every network call. No bare `except:`.
- Use logging, not print, in src/. Type hints + docstrings on public functions. Absolute paths via
  Path(__file__), not cwd-relative.
- Keep the existing dark theme/look; improve it, do not redesign from scratch.

# PHASE 0 - Hygiene & setup
Do the file moves/deletions above, fix .gitignore (do not ignore config_template.yaml or data/sample),
complete requirements.txt (keep transformers==4.33.0 pinned unless it fails to install, then tell me),
add config_template.yaml + .env.example, create tests/ and a pytest setup.
DONE WHEN: `pip install -r requirements.txt` succeeds and `pytest` runs (even with 0 tests).

# PHASE 1 - Training-side correctness
- prepare_dataset.py: parse data/Sentences_AllAgree.txt (latin-1, rsplit on the LAST "@") into
  data/raw/financial_phrasebank_clean.csv (columns: sentence,label).
- preprocess.py: no side effects at import; `main()` under __main__. Two functions: `light_clean` (strip
  HTML/URLs/extra whitespace only; keep case and stopwords; used for BERT) and `classic_clean` (for TF-IDF
  models; keep negation/direction words: no, not, nor, up, down, over, under, above, below, off).
- split_data.py: stratified 70/15/15, seed 42, saving columns text (light), text_classic, label.
- baseline.py reads text_classic; transformer.py reads text. In transformer.py set id2label/label2id
  from the label encoder, set save_total_limit=1, and WRITE metrics.json (accuracy, precision, recall,
  macro and weighted F1, n_test, trained_at) next to the saved model, plus append to model_metrics.csv.
- Add a small evaluation helper `python -m src.models.evaluate` that scores the current inference path on
  test.csv, so heuristic choices can be compared with real numbers.
DONE WHEN: tests prove light_clean keeps "not" and "down", the dataset parser returns 2264 rows
(1391 neutral / 570 positive / 303 negative), and the split preserves class ratios.
(Do NOT run the full training unless I ask; give me the exact commands and expected runtime instead.)

# PHASE 2 - Backend core (split the app.py monolith)
New modules under src/: config.py, news.py, prices.py, sentiment.py, insight.py, demo.py, pipeline.py.
- config.py: defaults < config.yaml < env vars (NEWSAPI_KEY, MODEL_PATH, DEMO_MODE). Companies live in
  config as {name, symbol, query, etf} (e.g. query '"Apple" AND (stock OR shares OR earnings)').
  Thresholds in config: flat_threshold_pct (default 0.5), price_window_hours (default 4), refresh_hours (2).
  Warn at startup if companies * (24/refresh_hours) exceeds 100 NewsAPI calls/day.
- news.py: NewsAPI fetch with header auth, timeout, retry on 5xx, handles 401/429, dedupes titles,
  strips " - Source" suffixes, caches raw JSON to data/raw/newsapi/ and falls back to the latest cache.
- sentiment.py: SentimentEngine that lazy-loads once, batch inference with torch.no_grad, label map from
  model config (fallback to config), reports available/unavailable instead of crashing. Local fine-tuned
  model first; optional hub fallback to ProsusAI/finbert only if config allows. The old ad-hoc heuristics
  (neutral floor, VADER assist) become OPTIONAL config flags, default OFF.
- prices.py: pure function reaction_from_history(hist, published_dt, window_hours) -> before_close,
  after_close, change_pct, graph_timestamps, graph_closes, or None. Handle tz-aware/naive indexes, hourly
  and daily bars, weekends/market closed, and missing data. Fetch history once per symbol per run.
  If the ticker has no usable data, retry with its sector ETF and tag the row proxy="<ETF>".
- insight.py (pure functions): suggestion matrix
    Positive+up -> "Consider Buy"; Negative+down -> "Consider Sell";
    Positive+down or Negative+up -> "Watch - possible delayed reaction";
    Neutral/Mixed or |change| < flat_threshold -> "Hold"; no price data -> "Hold - no price data".
  is_correct(): positive<->up, negative<->down, neutral<->flat; Mixed and no-price rows excluded.
  aggregate_accuracy(): overall + per company, returning percent, correct, total.
- pipeline.py: run_pipeline(cfg, engine, news_fn, history_fn) with injectable dependencies; scores each
  headline ONCE; writes data/processed/realtime/results.json (generated_at, source: live|cache|demo,
  metrics, companies summary, rows) and a CSV export. Row fields must include everything dashboard.html
  uses (rename change_24h -> change_pct in the template since it is not a 24h change) plus correct,
  proxy, price_available.
- demo.py: deterministic (seeded) synthetic rows for all companies with a realistic mix of correct and
  incorrect predictions, every headline marked as demo.
DONE WHEN: pytest passes with tests covering every suggestion branch, price edge cases (weekend, tz,
daily vs hourly, missing data, ETF fallback), pipeline with stubbed news/prices/sentiment, and demo output.

# PHASE 3 - Flask app (src/api/app.py becomes a thin app factory)
- create_app(config=None, engine=None, pipeline_fn=None) for testability. Run with `python -m src.api.app`
  and keep `python src/api/app.py` working.
- Routes: / (pass top 6 rows so the home preview works), /dashboard, /about, /api/results,
  /api/summary (per-company), /api/model_metrics, POST /api/predict (accepts {"text"} or {"texts"},
  max 50 items x 2000 chars, 503 with a clear message if the model is unavailable), POST /refresh
  (202 started / 409 already running), /health (model status, last run, source, last error, next run).
- Scheduler: start exactly once (guard the reloader; env flag for gunicorn, document --workers 1), a
  threading lock so runs never overlap, pipeline runs in a background thread, never blocks requests.
- Startup behavior: serve the last saved results immediately. If there are none and no API key, generate
  demo results. If live refresh fails, keep serving the last results and surface the error in the UI.
- Remove debug=True from the default run path (config flag only).
DONE WHEN: Flask test-client tests pass for every route, including refresh 409, predict validation,
and the no-model / no-key / network-failure paths.

# PHASE 4 - Interface (this is the visible part; make it excellent)
Keep the dark theme and Bootstrap/Chart.js stack. Required:
1. Dashboard header: "Last updated", next scheduled run, data source badge (LIVE / CACHED / DEMO),
   a "Refresh now" button with spinner and success/error toast (calls POST /refresh, then polls /health).
2. Persistent banners: DEMO DATA (amber), "Showing cached data - last refresh failed: <reason>" (red),
   "Sentiment model not loaded" (grey, with the fix command).
3. KPI strip driven by results.json: overall prediction accuracy (n correct of n), mismatches, articles
   analysed, companies tracked, and the active model name + test accuracy from metrics.json (or "n/a").
4. Per-company overview cards (new): sentiment mix bar, average confidence, average price move, accuracy,
   and the latest suggestion. Clicking a card filters the article list.
5. Article cards: sentiment badge with a text label (never color alone), confidence bar, before/after price
   with the window length ("price at publish -> +4h"), % change, suggestion pill, a correct/incorrect
   marker with tooltip, a "proxy: XLK" tag when the ETF fallback was used, sparkline, and a details modal.
6. Filters: sentiment, company, suggestion, correctness, text search; sort by time/confidence/price move;
   remember the chosen theme in localStorage.
7. Price/sentiment chart per company (Chart.js), plus a sentiment-distribution chart; both handle empty data.
8. Model page/section: render models/model_metrics.csv as a sortable table + bar chart (accuracy, macro-F1);
   highlight the best row from the data, not from hard-coded text. Missing file -> a clear empty state.
9. Replace every hardcoded accuracy/claim: sidebar, index.html stats bar and comparison grid, about.html
   (numbers from model_info; remove or config-drive the "Deployed on Render / HuggingFace Hub / ~4%" claims
   unless I confirm they are true). Company count and model count come from data.
10. States: loading skeletons, empty state, error state, and a no-JS-chart fallback (tables still readable).
11. Responsive (360 / 768 / 1280 px), keyboard-focusable controls, ARIA labels on icon buttons,
    contrast >= 4.5:1, rel="noopener noreferrer" on external links, footer disclaimer:
    "Heuristic signals for exploration, not financial advice."
If you can open a browser, load /, /dashboard and /about in demo mode at 360 and 1280 px and attach
screenshots; otherwise list exactly what I should check by eye.
DONE WHEN: all three pages render with zero console errors in demo mode, a live-mode run with a stubbed
pipeline updates the page after Refresh, and no hardcoded metric remains (grep for 90.03, 92.14, 86.71 to prove it).

# PHASE 5 - Docs & presentation support
- README rewritten to match what the code actually does: setup, config, demo mode, training commands,
  run commands, architecture (Mermaid), screenshots placeholder, known limitations.
- Known limitations to state plainly: NewsAPI free tier (about 100 requests/day, ~24h article delay,
  localhost only), yfinance intraday limits, sentiment-vs-price direction is a weak evaluation metric
  that ignores confounders and market-wide moves, small sample sizes, suggestions are not financial advice.
- CHANGES.md: before/after for each numbered problem above.
- 10-slide outline, a 3-minute demo script (run it in demo mode), and 10 likely examiner questions with
  honest answers (e.g. why was BERT trained on uncleaned text, how is accuracy defined, what are the
  limits of this evaluation).

# RULES FOR EVERY PHASE
- Before coding: list the files you will touch (5 lines max). After coding: run tests/commands, paste output,
  fix failures. If something cannot be verified in this environment (real FinBERT weights, live NewsAPI,
  yfinance network access), say so explicitly and give me the command to verify it myself.
- Never move to the next phase until I reply "next".

# FIRST RESPONSE
Reply with a table confirming which of the 14 problems you reproduced (and any you could not), any extra
issues you found, your assumptions, and at most 2 blocking questions. Then start Phase 0.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-06T18:08:13+05:30.

The user's current state is as follows:
Active Document: c:\Users\nikhi\Downloads\SentiLytics-main\SentiLytics-main\src\models\transformer.py (LANGUAGE_PYTHON)
Cursor is on line: 1
Other open documents:
- c:\Users\nikhi\Downloads\SentiLytics-main\SentiLytics-main\src\models\transformer.py (LANGUAGE_PYTHON)
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Claude Opus 4.6 (Thinking). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>