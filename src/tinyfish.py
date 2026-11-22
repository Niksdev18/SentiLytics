"""TinyFish API client module for web search, content scraping, and agent automation.

Features:
- Search API: GET https://api.search.tinyfish.ai?query=...
- Fetch API: POST https://api.fetch.tinyfish.ai (extract markdown/text from URLs)
- Agent API: POST https://agent.tinyfish.ai/v1/automation/run (goal-directed web automation)
- Header authentication via X-API-Key
- Configurable timeout and retries
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai"
TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"
TINYFISH_AGENT_URL = "https://agent.tinyfish.ai/v1/automation/run"


class TinyFishClient:
    """Client for TinyFish web infrastructure and agent automation APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def is_configured(self) -> bool:
        """Check if API key is provided and non-empty."""
        return bool(self.api_key and self.api_key.strip())

    def _get_headers(self) -> Dict[str, str]:
        """Construct request headers with API key authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key.strip()
        return headers

    def search(
        self,
        query: str,
        page_size: int = 5,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Search the web via TinyFish Search API.

        Args:
            query: The search term or boolean query.
            page_size: Max number of results.
            timeout: Request timeout in seconds.

        Returns:
            Dict containing 'results' list and query info.
        """
        if not self.is_configured:
            logger.warning("TinyFish API key is not configured.")
            return {"results": [], "query": query, "error": "API key not configured"}

        req_timeout = timeout or self.timeout
        params = {"query": query}
        headers = self._get_headers()

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(
                    TINYFISH_SEARCH_URL,
                    headers=headers,
                    params=params,
                    timeout=req_timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                        data["results"] = data["results"][:page_size]
                    return data
                elif resp.status_code in (401, 403):
                    logger.warning(
                        "TinyFish authentication failed (%s): %s",
                        resp.status_code,
                        resp.text[:100],
                    )
                    return {"results": [], "query": query, "error": f"Auth error {resp.status_code}"}
                elif resp.status_code == 429:
                    logger.warning("TinyFish rate limit reached (429).")
                    return {"results": [], "query": query, "error": "Rate limited"}
                elif resp.status_code >= 500:
                    logger.warning(
                        "TinyFish 5xx server error (%s) on attempt %d",
                        resp.status_code,
                        attempt + 1,
                    )
                    if attempt < self.max_retries:
                        time.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning("TinyFish unexpected status %s: %s", resp.status_code, resp.text[:100])
                    break
            except (requests.RequestException, TimeoutError) as e:
                logger.warning("Network error calling TinyFish Search: %s", e)
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        return {"results": [], "query": query, "error": "Failed after retries"}

    def fetch_content(
        self,
        urls: List[str],
        format_type: str = "markdown",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Fetch cleaned full-page content/markdown via TinyFish Fetch API.

        Args:
            urls: List of web URLs to render and extract.
            format_type: Output format: 'markdown', 'json', or 'html'.
            timeout: Request timeout in seconds.

        Returns:
            Dict containing 'results' list and 'errors' list.
        """
        if not self.is_configured:
            return {"results": [], "errors": ["API key not configured"]}

        req_timeout = timeout or self.timeout
        headers = self._get_headers()
        payload = {"urls": urls, "format": format_type}

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    TINYFISH_FETCH_URL,
                    headers=headers,
                    json=payload,
                    timeout=req_timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code in (401, 403, 429):
                    logger.warning("TinyFish fetch status %s: %s", resp.status_code, resp.text[:100])
                    return {"results": [], "errors": [f"Status {resp.status_code}"]}
                elif resp.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                logger.warning("Error fetching content via TinyFish: %s", e)
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        return {"results": [], "errors": ["Failed after retries"]}

    def run_agent(
        self,
        url: str,
        goal: str,
        timeout: Optional[float] = 30.0,
    ) -> Dict[str, Any]:
        """Run an automated TinyFish browser agent with a natural language goal.

        Args:
            url: Target website URL.
            goal: Plain-English task description for the agent.
            timeout: Execution timeout in seconds.

        Returns:
            Dict containing agent result or error.
        """
        if not self.is_configured:
            return {"error": "API key not configured"}

        headers = self._get_headers()
        payload = {"url": url, "goal": goal}

        try:
            resp = requests.post(
                TINYFISH_AGENT_URL,
                headers=headers,
                json=payload,
                timeout=timeout or 30.0,
            )
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Status {resp.status_code}", "details": resp.text[:200]}
        except Exception as e:
            logger.warning("Error running TinyFish Agent: %s", e)
            return {"error": str(e)}
