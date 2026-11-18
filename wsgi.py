"""WSGI entrypoint for production deployment (Gunicorn / uWSGI).

Ensures the Flask application factory is instantiated properly with
production settings. In multi-process production environments, use
`--workers 1` so that APScheduler and in-memory application state
remain singletons without duplicated background fetch jobs.
"""

import os
from src.api.app import create_app

# Start the background scheduler only if enabled (default True)
start_scheduler = os.environ.get("START_SCHEDULER", "true").lower() in ("true", "1")

# Create WSGI application instance
app = create_app(start_scheduler=start_scheduler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 5000)))
    app.run(host="0.0.0.0", port=port)
