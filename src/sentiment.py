"""Sentiment Analysis Engine using FinBERT with batch inference and fallback support.

Features:
- Lazy model loading (loaded once in memory)
- Batch inference with torch.no_grad()
- Label mapping dynamically read from model config (with robust fallback)
- Local fine-tuned model loaded first; optional fallback to ProsusAI/finbert Hub
- Reports available/unavailable status without crashing
- Config-driven optional heuristics (neutral floor, VADER assist), default OFF
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import torch

from src.config import AppConfig

logger = logging.getLogger(__name__)


class SentimentEngine:
    """Manages FinBERT model loading and batch sentiment scoring."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.status: str = "uninitialized"  # 'ready', 'unavailable', 'uninitialized'
        self.model_description: str = "None"
        self.is_fine_tuned: bool = False
        self.id2label: Dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self._vader_analyzer: Optional[Any] = None

    def _normalize_label(self, raw_label: Any) -> str:
        """Normalize label string to standard TitleCase."""
        s = str(raw_label).strip().lower()
        if "pos" in s:
            return "Positive"
        elif "neg" in s:
            return "Negative"
        return "Neutral"

    def load(self) -> bool:
        """Load transformer model: local first, then Hub fallback if allowed."""
        root = Path(__file__).resolve().parent.parent
        if Path(self.config.model_path).is_absolute():
            model_path = Path(self.config.model_path)
        else:
            model_path = root / self.config.model_path

        # 1. Try local fine-tuned model
        if model_path.exists() and (model_path / "config.json").exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                logger.info("Loading local transformer model from %s", model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                self.model.to(self.device)
                self.model.eval()

                # Read id2label from config
                if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                    self.id2label = {
                        int(k): self._normalize_label(v) for k, v in self.model.config.id2label.items()
                    }

                # Check if metrics.json exists
                metrics_file = model_path / "metrics.json"
                if metrics_file.exists():
                    self.is_fine_tuned = True
                    self.model_description = "FinBERT (fine-tuned)"
                else:
                    self.model_description = "FinBERT (local)"

                self.status = "ready"
                return True
            except Exception as e:
                logger.warning("Failed to load local model from %s: %s", model_path, e)

        # 2. Try Hugging Face Hub fallback
        if self.config.allow_hub_fallback:
            hub_name = "ProsusAI/finbert"
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                logger.info("Loading Hub model %s", hub_name)
                self.tokenizer = AutoTokenizer.from_pretrained(hub_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(hub_name)
                self.model.to(self.device)
                self.model.eval()

                if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                    self.id2label = {
                        int(k): self._normalize_label(v) for k, v in self.model.config.id2label.items()
                    }

                self.is_fine_tuned = False
                self.model_description = "FinBERT (pretrained)"
                self.status = "ready"
                return True
            except Exception as e:
                logger.warning("Failed to load Hub model %s: %s", hub_name, e)

        # 3. Model unavailable
        self.status = "unavailable"
        self.model_description = "Model Unavailable"
        logger.warning("Sentiment engine unavailable — no local model and Hub fallback disabled or failed.")
        return False

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run batch inference on a list of raw text headlines.

        Args:
            texts: List of strings.

        Returns:
            List of dicts with 'label', 'confidence', 'probabilities'.
        """
        if not texts:
            return []

        # Lazy load if needed
        if self.status == "uninitialized":
            self.load()

        if self.status != "ready" or self.model is None or self.tokenizer is None:
            # Fallback to VADER when transformer is unavailable
            return self._vader_fallback_batch(texts)

        results = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1).cpu()

            for j, p in enumerate(probs):
                p_list = p.tolist()
                prob_dict = {
                    self.id2label.get(k, f"Class_{k}"): round(float(val), 4)
                    for k, val in enumerate(p_list)
                }

                pred_idx = int(torch.argmax(p).item())
                pred_label = self.id2label.get(pred_idx, "Neutral")
                confidence = round(float(p[pred_idx].item()), 4)

                # Heuristic 1: Neutral Floor (optional, default off)
                if self.config.neutral_floor_enabled and pred_label == "Neutral":
                    neu_prob = prob_dict.get("Neutral", 0.0)
                    if neu_prob < self.config.neutral_floor_threshold:
                        pos_p = prob_dict.get("Positive", 0.0)
                        neg_p = prob_dict.get("Negative", 0.0)
                        if pos_p > neg_p:
                            pred_label = "Positive"
                            confidence = pos_p
                        else:
                            pred_label = "Negative"
                            confidence = neg_p

                # Heuristic 2: VADER assist (optional, default off)
                if self.config.vader_assist_enabled:
                    vader_label = self._vader_score(chunk[j])
                    if pred_label == "Neutral" and vader_label != "Neutral":
                        pred_label = vader_label

                results.append({
                    "label": pred_label,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                })

        return results

    def _vader_score(self, text: str) -> str:
        """Compute VADER sentiment score."""
        if self._vader_analyzer is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
        c = self._vader_analyzer.polarity_scores(text)["compound"]
        if c >= 0.05:
            return "Positive"
        elif c <= -0.05:
            return "Negative"
        return "Neutral"

    def _vader_fallback_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback when transformer model is not loaded."""
        results = []
        for text in texts:
            lbl = self._vader_score(text)
            results.append({
                "label": lbl,
                "confidence": 0.50,
                "probabilities": {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34},
            })
        return results
