# Senti-Lytics: Presentation Deck, Demo Script & Examiner Q&A

This guide prepares you for your project presentation and viva examination. It provides a slide-by-slide presentation outline, a concise 3-minute live demo script, and honest, technically grounded answers to tough examiner questions.

---

## 🖥️ 10-Slide Presentation Outline

### Slide 1: Title & Motivation
- **Title**: Senti-Lytics: Real-Time Financial Sentiment Analysis and Market Reaction Tracking
- **Subtitle**: Closing the gap between NLP sentiment inference and short-term market price dynamics
- **Presenter**: [Your Name]
- **Key Hook**: Financial headlines influence intraday volatility, but how well does model sentiment correlate with immediate market reactions? Senti-Lytics builds an automated, verifiable pipeline to ingest, classify, track, and evaluate financial news sentiment.

### Slide 2: The Core Problem & Research Objectives
- **Challenges in Financial NLP**:
  - Domain-specific jargon (e.g., "liability", "bullish", "hedging" have non-colloquial meanings).
  - High class imbalance in financial corpora.
  - Naive evaluation: High accuracy on static test sets often fails to produce actionable insights in live market contexts.
- **Objectives**:
  1. Build a robust pipeline from raw news ingestion to stock price reaction tracking.
  2. Evaluate domain-specific transformer models (FinBERT) against tuned classical baselines.
  3. Formulate transparent heuristic trading suggestions (Buy, Sell, Watch, Hold) and measure directional agreement empirically.

### Slide 3: Architecture & System Overview
- **Visual**: End-to-end pipeline architecture (Ingestion $\rightarrow$ Preprocessing $\rightarrow$ FinBERT Inference $\rightarrow$ Market Pricing $\rightarrow$ Insight Engine $\rightarrow$ Flask Dashboard).
- **Core Engineering Decisions**:
  - Decoupled modules with injectable interfaces.
  - Background scheduler (APScheduler) protected by threading locks to prevent duplicate execution.
  - Layered configuration and deterministic DEMO mode for resilient offline demonstrations.

### Slide 4: Data Engineering & Text Preprocessing
- **Dataset**: Financial PhraseBank (2,264 sentences with 100% annotator agreement).
  - Class distribution: 1,391 Neutral (61.4%), 570 Positive (25.2%), 303 Negative (13.4%).
- **Addressing the Train/Serve Text Mismatch**:
  - *Standard NLP trap*: Aggressive stopword removal strips "not", "no", "up", "down", destroying critical sentiment polarity.
  - *Light Cleaning (Transformers)*: Preserves capitalization, grammatical structure, and punctuation; strips only formatting artifacts.
  - *Classical Cleaning (TF-IDF)*: Preserves negation tokens (`no`, `not`, `nor`) and directional prepositions (`up`, `down`, `above`, `below`).

### Slide 5: Model Development & Benchmarking
- **Model Hierarchy**:
  - Classical Baselines: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest (evaluated with TF-IDF 1-3 grams, balanced class weighting).
  - Deep Learning: Pretrained BERT (`bert-base-uncased`) vs Domain-Adapted FinBERT (`ProsusAI/finbert`).
- **Benchmark Findings**:
  - FinBERT achieves superior macro-F1 on nuanced financial phrasing compared to generic BERT and classical models.
  - Linear SVM and Logistic Regression provide strong, low-latency baselines for high-throughput scenarios.

### Slide 6: Market Reaction Alignment
- **The "Before/After" Window**:
  - Rather than using arbitrary daily close prices, Senti-Lytics computes intraday price reaction centered around each article's publication timestamp (`t_pub` to `t_pub + 4h`).
- **Edge-Case Engineering**:
  - Timezone harmonization (UTC vs exchange local time).
  - Weekend and overnight publication handling (aligned to next market opening bell).
  - Ticker fallback: If an individual equity lacks liquidity or quote data, the system falls back to its sector ETF (e.g., `XLK`) tagged transparently with `proxy="XLK"`.

### Slide 7: Suggestion Matrix & Directional Accuracy
- **Signal Logic**:
  - **Consider Buy**: Positive sentiment + stock price increased ($> +0.5\%$).
  - **Consider Sell**: Negative sentiment + stock price decreased ($< -0.5\%$).
  - **Watch (Delayed Reaction)**: Sentiment and price move in opposite directions (e.g., positive news but initial dip), flagging potential delayed market absorption.
  - **Hold**: Neutral sentiment or flat price movement ($|\Delta| \le 0.5\%$).
- **Directional Accuracy**: Measures the percentage of directional signals where market movement agreed with headline polarity.

### Slide 8: Live Dashboard & UX Design
- **Key Features**:
  - Responsive Dark-Theme UI built on Bootstrap 5 and Chart.js.
  - Live / Cached / Demo mode banners ensuring transparent data provenance.
  - Real-time KPI summary: Directional accuracy, article volume, positive/neutral/negative balance.
  - Interactive company cards, article detail modals, and historical benchmark comparison tables.

### Slide 9: Methodological Limitations (Honest Science)
- **NewsAPI Free-Tier Constraints**: 100 requests/day quota, ~24h delay on free tier articles, localhost restrictions.
- **Confounding Variables in Stock Prices**: Intraday prices respond to broader market beta, macroeconomic indicators (Fed interest rate announcements, CPI), and index movements, not isolated headlines.
- **Sample Size**: Financial PhraseBank is modest in size (2,264 sentences); generalizability to complex regulatory filings (10-K, 10-Q) requires long-document models.
- **Ethical Disclaimer**: Heuristic research prototype, not financial advice.

### Slide 10: Conclusion & Future Work
- **Summary**: Delivered a fully verified, test-covered (49 unit tests) financial analytics platform with verified data integrity and honest metrics.
- **Future Directions**:
  - Multi-source news ingestion (SEC EDGAR filings, Bloomberg/Reuters RSS).
  - Cross-market beta neutralization (evaluating residual price movement after removing S&P 500 drift).
  - Temporal event-study windows using high-frequency tick data.

---

## ⏱️ 3-Minute Live Demo Script

### Target Duration: 3 Minutes
*Prerequisite: Flask server running in Demo Mode (`DEMO_MODE=True python -m src.api.app`)*

| Time | Screen / Action | Spoken Script |
|------|-----------------|---------------|
| **0:00 - 0:30** | **Home Page (`/`)** | "Good morning. I'm presenting **Senti-Lytics**, a real-time financial sentiment analysis system that tracks headline sentiment against actual stock price movement. On the home page, you immediately see our tracked portfolio—including Apple, Microsoft, Google, and Amazon—along with quick previews of recent sentiment classifications." |
| **0:30 - 1:15** | **Dashboard (`/dashboard`) - KPI & Banners** | "Moving to the main dashboard. Notice the prominent **DEMO DATA** banner—in adherence to scientific honesty, our interface always shows the exact data provenance (Live, Cached, or Demo). In the top KPI strip, we see our aggregate directional accuracy across 24 analyzed articles, our sentiment breakdown, and the active FinBERT model. The cards below summarize sentiment and average price moves for each company." |
| **1:15 - 2:00** | **Article Cards & Suggestion Matrix** | "Let's look at an individual article card. Here is a headline for Microsoft: *'Microsoft beats quarterly cloud revenue estimates'*. FinBERT classified this as **Positive** with 94% confidence. Our market engine tracked Microsoft's price from publication over a 4-hour window: the stock moved up +1.24%. Because positive sentiment aligned with price appreciation, our suggestion engine issued a **Consider Buy** signal, marked with a green 'Correct Direction' badge. Conversely, when news is positive but price dips initially, the system flags **'Watch - possible delayed reaction'**." |
| **2:00 - 2:30** | **Live Refresh & API** | "If we click **Refresh Now**, the background scheduler asynchronously polls the pipeline without blocking user interaction. Behind the scenes, we enforce strict rate limiting, 5xx retries, and header-based authentication for external APIs. We also expose a fully tested REST API at `/api/predict` for programmatic batch inference." |
| **2:30 - 3:00** | **Model Benchmark Section & Conclusion** | "Finally, scrolling to our Model Benchmark section: every figure displayed here is dynamically pulled from our reproducible training logs—never hardcoded. Senti-Lytics bridges the gap between text classification and market reality with full transparency. I'm ready for your questions." |

---

## 🎓 10 Likely Examiner Questions & Honest Answers

### Q1: Why did you train FinBERT on lightly cleaned text while using more aggressive cleaning for TF-IDF?
**Answer**:
"BERT and other transformer models rely on bidirectional self-attention across the full sentence. Word order, capitalization, punctuation, and function words—especially negations like 'not', 'no', and 'neither'—are critical for positional embeddings. Stripping stopwords would turn 'Profit did not fall' into 'Profit fall', completely reversing the sentiment. For classical TF-IDF models, which lack positional awareness and treat text as a bag-of-words, dimensionality reduction via stopword removal is helpful; however, we specifically preserved negation and financial direction tokens ('not', 'no', 'up', 'down', 'below') in our classical cleaner to prevent signal loss."

### Q2: How is "Directional Accuracy" defined in your system, and what are its statistical limitations?
**Answer**:
"Directional accuracy measures whether stock price movement matched the sentiment direction: positive news with price increase ($\Delta > +0.5\%$), negative news with price decrease ($\Delta < -0.5\%$), or neutral news with flat price ($|\Delta| \le 0.5\%$). Its main limitation is that correlation does not imply causation. A company's stock price fluctuates due to macroeconomic factors, interest rates, sector-wide momentum, and earnings releases of competitors—not solely the individual headline being analyzed. Therefore, directional accuracy is a heuristic exploration metric rather than proof of predictive causality."

### Q3: How do you handle class imbalance in Financial PhraseBank?
**Answer**:
"The Financial PhraseBank dataset is heavily skewed: over 60% of sentences are Neutral, while Positive represents ~25% and Negative only ~13%. We address this in two ways:
1. In transformer fine-tuning, we compute inverse class frequencies and apply weighted cross-entropy loss, penalizing errors on the minority Negative and Positive classes more heavily.
2. We evaluate all models primarily on **macro-averaged F1 score** and class-specific precision/recall rather than overall accuracy, which would be artificially inflated by the majority neutral class."

### Q4: Why use a 4-hour price window instead of standard daily close prices?
**Answer**:
"Daily close prices introduce severe temporal misalignment. If an article is published at 3:30 PM, measuring the change from yesterday's close to today's close mostly reflects what happened *before* the article was published. By using intraday hourly bars centered around the publication timestamp, we isolate the immediate market absorption window. If news arrives after market close or over a weekend, our pricing module aligns the window to start at the subsequent market open."

### Q5: What happens when market data is unavailable for a specific ticker?
**Answer**:
"In real-world financial feeds, smaller tickers may be delisted, halted, or unquoted in free APIs. When `yfinance` returns empty history for a ticker, our pricing module falls back to the corresponding sector ETF (for example, using `XLK` for technology equities or `XLF` for financials). Crucially, the system tags the resulting article card with `proxy='XLK'`, ensuring full transparency that the user is viewing sector-level proxy movement."

### Q6: How does your background scheduler prevent race conditions and API rate limit breaches?
**Answer**:
"We implement a background `threading.Lock` inside the application factory. When a scheduled run or manual `/refresh` is triggered, the worker acquires the lock. If another refresh is attempted while a job is running, the endpoint immediately returns HTTP 409 Conflict. Furthermore, our configuration validator calculates `companies * (24 / refresh_hours)` on startup to guarantee our daily volume remains well within NewsAPI's 100 requests/day free tier limit."

### Q7: Why not use VADER or custom keyword overrides to boost performance?
**Answer**:
"Early versions of the codebase used ad-hoc heuristics—such as overriding FinBERT with VADER or forcing neutral predictions below 70% into directional calls. During testing, we found these rules produced severe keyword bugs (for example, matching 'fine' inside 'refinance' or flipping sentiments due to distant negation words) and distorted model calibration. We removed these heuristics from the default inference path to maintain pure, statistically sound FinBERT probabilities."

### Q8: What is the purpose of the 'Watch - possible delayed reaction' signal?
**Answer**:
"Markets do not always react immediately or intuitively to news. For example, a company might release positive headline earnings, but the stock initially drops due to algorithmic profit-taking or high pre-announcement expectations. Rather than classifying this simply as a 'bad' prediction, our matrix flags it as 'Watch - possible delayed reaction', signaling to the analyst that price and sentiment diverge and warrant closer monitoring over a longer horizon."

### Q9: Could this system be deployed for actual live algorithmic trading today?
**Answer**:
"No, and we state this prominently in our interface disclaimer. Real algorithmic trading requires tick-level order book data, bid-ask spread accounting, transaction cost modeling, and multi-factor risk controls. NewsAPI's free tier has an approximate 24-hour publication delay, meaning headlines arrive well after market participants have already priced in the information. Senti-Lytics is designed as an analytical research dashboard, not an execution engine."

### Q10: How did you ensure testability across the entire pipeline without requiring GPU access or live network connections?
**Answer**:
"We structured the system around dependency injection and pure functions:
- `reaction_from_history` and `insight.py` logic are pure functions tested with synthetic time series.
- The pipeline accepts mock news fetchers and mock price fetchers.
- Model tests verify configuration schemas, tokenization, and label preservation without requiring multi-gigabyte downloads.
- Our test suite of 49 automated pytest tests runs in under 90 seconds on standard CPU hardware."

