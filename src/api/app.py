"""Flask Application Factory and Web Server.

Features:
- create_app() application factory for testing and production
- Non-blocking background news refresh with threading lock
- BackgroundScheduler with reloader protection
- Clean REST API: /api/results, /api/summary, /api/model_metrics, /api/predict, /refresh, /health
- Immediate startup serving cached results or demo fallback
- Production-safe: debug=False by default
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import json
import logging
import os
import threading
from flask import Flask, render_template, jsonify, request

from src.config import AppConfig, load_config
from src.demo import generate_demo_results
from src.pipeline import run_pipeline
from src.sentiment import SentimentEngine

logger = logging.getLogger(__name__)


class AppState:
    """Encapsulates runtime state, cache, and synchronization for the Flask app."""

    def __init__(
        self,
        config: AppConfig,
        engine: Optional[SentimentEngine] = None,
        pipeline_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.engine = engine or SentimentEngine(config)
        self.pipeline_fn = pipeline_fn or run_pipeline
        self.refresh_lock = threading.Lock()
        self.is_refreshing: bool = False
        self.last_error: Optional[str] = None
        self.last_refresh_time: Optional[str] = None
        self.next_run_time: Optional[str] = None
        self.results: Dict[str, Any] = {}

        # Load initial data on startup
        self.load_initial_results()

    def load_initial_results(self) -> None:
        """Load last saved results.json, or generate demo data if missing."""
        results_file = self.config.root_dir / "data" / "processed" / "realtime" / "results.json"
        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    self.results = json.load(f)
                    self.last_refresh_time = self.results.get("generated_at")
                    logger.info("Loaded previous results from %s", results_file)
                    return
            except Exception as e:
                logger.warning("Error reading %s: %s", results_file, e)

        # Fallback to demo mode if no previous results or no API key
        logger.info("No saved results found. Initializing with DEMO dataset.")
        self.results = generate_demo_results(self.config)
        self.last_refresh_time = self.results.get("generated_at")

    def run_refresh_task(self) -> None:
        """Background worker executing the data pipeline under a lock."""
        acquired = self.refresh_lock.acquire(blocking=False)
        if not acquired:
            return

        self.is_refreshing = True
        try:
            logger.info("Starting background pipeline refresh...")
            new_results = self.pipeline_fn(
                config=self.config,
                engine=self.engine,
            )
            if new_results and "rows" in new_results:
                self.results = new_results
                self.last_refresh_time = new_results.get("generated_at")
                self.last_error = None
                logger.info("Pipeline refresh completed successfully.")
            else:
                self.last_error = "Pipeline returned empty results"
        except Exception as e:
            logger.error("Pipeline refresh failed: %s", e)
            self.last_error = str(e)
        finally:
            self.is_refreshing = False
            self.refresh_lock.release()


def create_app(
    config: Optional[AppConfig] = None,
    engine: Optional[SentimentEngine] = None,
    pipeline_fn: Optional[Callable] = None,
    start_scheduler: bool = False,
) -> Flask:
    """Flask application factory."""
    cfg = config or load_config()

    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))

    state = AppState(config=cfg, engine=engine, pipeline_fn=pipeline_fn)
    app.state = state  # type: ignore

    # Optional background scheduler
    if start_scheduler:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            scheduler = BackgroundScheduler()
            interval_hours = max(1, cfg.refresh_hours)
            scheduler.add_job(state.run_refresh_task, "interval", hours=interval_hours)
            scheduler.start()
            next_run = datetime.now(timezone.utc) + timedelta(hours=interval_hours)
            state.next_run_time = next_run.isoformat()
            app.scheduler = scheduler  # type: ignore
        except Exception as e:
            logger.warning("Failed to start BackgroundScheduler: %s", e)

    # ─────────────────────────────────────────────────────────
    # Page Routes
    # ─────────────────────────────────────────────────────────
    @app.route("/")
    def home():
        rows = state.results.get("rows", [])
        for r in rows:
            if "scraped_at" not in r:
                r["scraped_at"] = r.get("published_at", "")
        top_news = rows[:6] if rows else []
        return render_template(
            "index.html",
            news=top_news,
            metrics=state.results.get("metrics"),
            companies=state.results.get("companies", {}),
            source=state.results.get("source", "demo"),
            model_name=state.results.get("model_name", "FinBERT"),
            model_metrics=_get_model_metrics(state.config),
            generated_at=state.results.get("generated_at"),
            last_error=state.last_error,
        )

    @app.route("/dashboard")
    def dashboard():
        metrics = state.results.get("metrics")
        if metrics and isinstance(metrics, dict):
            if "accuracy" not in metrics:
                metrics["accuracy"] = metrics.get("accuracy_pct", 0.0)
            if "correct_predictions" not in metrics:
                metrics["correct_predictions"] = metrics.get("correct", 0)
            if "total_predictions" not in metrics:
                metrics["total_predictions"] = metrics.get("total", 0)

        rows = state.results.get("rows", [])
        for r in rows:
            if "scraped_at" not in r:
                r["scraped_at"] = r.get("published_at", "")

        return render_template(
            "dashboard.html",
            news=rows,
            metrics=metrics,
            model_metrics=_get_model_metrics(state.config),
            companies=state.results.get("companies", {}),
            source=state.results.get("source", "demo"),
            generated_at=state.results.get("generated_at"),
            last_error=state.last_error,
            model_name=state.results.get("model_name", "FinBERT"),
        )

    @app.route("/about")
    def about():
        return render_template(
            "about.html",
            model_metrics=_get_model_metrics(state.config),
            model_name=state.results.get("model_name", "FinBERT"),
        )

    # ─────────────────────────────────────────────────────────
    # API Routes
    # ─────────────────────────────────────────────────────────
    @app.route("/api/results", methods=["GET"])
    def api_results():
        return jsonify(state.results)

    @app.route("/api/summary", methods=["GET"])
    def api_summary():
        return jsonify(state.results.get("companies", {}))

    @app.route("/api/model_metrics", methods=["GET"])
    def api_model_metrics():
        metrics_data = _get_model_metrics(state.config)
        return jsonify(metrics_data)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON body. Provide {'text': '...'} or {'texts': [...]}"}), 400

        texts: List[str] = []
        if "text" in data and isinstance(data["text"], str):
            texts = [data["text"]]
        elif "texts" in data and isinstance(data["texts"], list):
            texts = [str(t) for t in data["texts"]]
        else:
            return jsonify({"error": "Payload must contain 'text' string or 'texts' list"}), 400

        # Validation: max 50 items, max 2000 chars per item
        if len(texts) == 0:
            return jsonify({"error": "No text provided for analysis"}), 400
        if len(texts) > 50:
            return jsonify({"error": f"Maximum 50 items allowed per request, received {len(texts)}"}), 400
        for idx, t in enumerate(texts):
            if len(t) > 2000:
                return jsonify({"error": f"Item {idx} exceeds 2000 characters limit"}), 400

        # Check engine availability
        if state.engine.status == "uninitialized":
            state.engine.load()

        if state.engine.status == "unavailable" and not state.config.allow_hub_fallback:
            return jsonify({
                "error": "Sentiment model is currently unavailable",
                "message": "Run training or enable allow_hub_fallback in config.yaml",
            }), 503

        try:
            preds = state.engine.predict_batch(texts)
            return jsonify({
                "model": state.engine.model_description,
                "count": len(preds),
                "predictions": preds,
            })
        except Exception as e:
            logger.error("Inference error: %s", e)
            return jsonify({"error": "Inference failed", "details": str(e)}), 500

    @app.route("/refresh", methods=["POST"])
    def refresh():
        if state.is_refreshing:
            return jsonify({"status": "conflict", "message": "Refresh already running"}), 409

        # Trigger refresh in background thread
        thread = threading.Thread(target=state.run_refresh_task, daemon=True)
        thread.start()
        return jsonify({"status": "accepted", "message": "Refresh started"}), 202

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model_status": state.engine.status,
            "model_name": state.engine.model_description,
            "is_fine_tuned": state.engine.is_fine_tuned,
            "is_refreshing": state.is_refreshing,
            "source": state.results.get("source", "none"),
            "last_refresh": state.last_refresh_time,
            "last_error": state.last_error,
            "next_run": state.next_run_time,
        })

    return app


def _get_model_metrics(cfg: AppConfig) -> Dict[str, Any]:
    """Retrieve real training metrics or return a clear unavailable notice."""
    root = cfg.root_dir
    metrics_json = root / "models" / "transformer_model" / "best_model" / "metrics.json"

    if metrics_json.exists():
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Error reading metrics.json: %s", e)

    # Fallback to model_metrics.csv if available
    metrics_csv = root / "models" / "model_metrics.csv"
    if metrics_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            return {
                "status": "Available in model_metrics.csv",
                "models": df.to_dict(orient="records"),
            }
        except Exception as e:
            logger.warning("Error reading model_metrics.csv: %s", e)

    return {"status": "Not available - run training"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app(start_scheduler=True)
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    print(f"Starting SentiLytics server on http://127.0.0.1:{port} (debug={debug_mode})")
    app.run(host="127.0.0.1", port=port, debug=debug_mode, use_reloader=False)