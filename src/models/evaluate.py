"""Evaluation helper to score inference models on test.csv.

Enables comparison of raw predictions vs heuristic modifications (neutral floor, VADER assist)
against ground-truth test labels.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

logger = logging.getLogger(__name__)


def evaluate_test_set(
    test_csv: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    use_heuristics: bool = False,
    neutral_threshold: float = 0.70,
) -> Dict[str, Any]:
    """Score model or heuristic pipeline on test dataset.

    Args:
        test_csv: Path to test.csv. Defaults to data/processed/test.csv.
        model_dir: Path to model directory (e.g., models/transformer_model/best_model).
        use_heuristics: Whether to apply neutral confidence floor / heuristics.
        neutral_threshold: Threshold below which neutral prediction is modified.

    Returns:
        Dict of computed metrics (accuracy, macro_f1, weighted_f1, report).
    """
    root = Path(__file__).resolve().parent.parent.parent
    if test_csv is None:
        test_csv = root / "data" / "processed" / "test.csv"
    if model_dir is None:
        model_dir = root / "models" / "transformer_model" / "best_model"

    if not test_csv.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_csv}. Run split_data.py first.")

    df = pd.read_csv(test_csv).dropna(subset=["text", "label"])
    y_true = df["label"].astype(str).str.lower().tolist()
    texts = df["text"].astype(str).tolist()

    # Check if a model exists
    if model_dir.exists() and (model_dir / "config.json").exists():
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        y_preds = []

        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
                pred_idx = int(torch.argmax(probs).item())
                pred_label = id2label.get(pred_idx, "neutral")

                # Optional heuristic: neutral floor override
                if use_heuristics and pred_label == "neutral":
                    neu_idx = [i for i, l in id2label.items() if l == "neutral"]
                    if neu_idx and probs[neu_idx[0]].item() < neutral_threshold:
                        # force highest non-neutral prob
                        probs_non_neu = probs.clone()
                        probs_non_neu[neu_idx[0]] = -1.0
                        forced_idx = int(torch.argmax(probs_non_neu).item())
                        pred_label = id2label.get(forced_idx, pred_label)

                y_preds.append(pred_label)
    else:
        # Fallback to zero-shot/lexicon evaluation (VADER baseline) for immediate testing
        logger.info("No trained transformer found at %s. Using VADER as baseline evaluation.", model_dir)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        y_preds = []
        for text in texts:
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                pred = "positive"
            elif compound <= -0.05:
                pred = "negative"
            else:
                pred = "neutral"
            y_preds.append(pred)

    acc = accuracy_score(y_true, y_preds)
    macro_f1 = f1_score(y_true, y_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_preds, average="weighted", zero_division=0)
    report = classification_report(y_true, y_preds, zero_division=0)

    results = {
        "n_samples": len(y_true),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "report": report,
    }
    return results


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate inference on test.csv")
    parser.add_argument("--test-file", type=str, default=None, help="Path to test.csv")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to model directory")
    parser.add_argument("--use-heuristics", action="store_true", help="Apply neutral floor heuristic")
    args = parser.parse_args()

    test_path = Path(args.test_file) if args.test_file else None
    model_path = Path(args.model_dir) if args.model_dir else None

    results = evaluate_test_set(
        test_csv=test_path,
        model_dir=model_path,
        use_heuristics=args.use_heuristics,
    )
    print("\n--- Test Set Evaluation ---")
    print(f"Evaluated {results['n_samples']} samples")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Macro F1:    {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print("\nClassification Report:\n", results["report"])


if __name__ == "__main__":
    main()
