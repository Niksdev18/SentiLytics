"""Transformer (FinBERT) fine-tuning pipeline.

Reads 'text' (lightly cleaned) from data/processed/train.csv, val.csv, and test.csv.
Configures id2label and label2id in the model config, sets save_total_limit=1,
and writes comprehensive metrics.json next to the saved model artifact.
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class NewsDataset(Dataset):
    """PyTorch Dataset for text tokenization."""

    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred) -> Dict[str, float]:
    """Compute evaluation metrics during training."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec_w,
        "recall": rec_w,
        "f1": f1_w,
        "macro_f1": f1_m,
    }


def train_transformer(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_name: str = "ProsusAI/finbert",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Dict[str, Any]:
    """Fine-tune FinBERT model on lightly cleaned financial news text.

    Args:
        data_dir: Path to directory containing train.csv, val.csv, test.csv.
        output_dir: Path to directory where model and metrics should be saved.
        model_name: HuggingFace model checkpoint.
        epochs: Training epochs.
        batch_size: Batch size for train/eval.
        learning_rate: Learning rate for AdamW optimizer.

    Returns:
        Dict with test evaluation metrics.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if data_dir is None:
        data_dir = root / "data" / "processed"
    if output_dir is None:
        output_dir = root / "models" / "transformer_model"

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"

    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Datasets not found in {data_dir}. Run split_data.py first.")

    train_df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    val_df = pd.read_csv(val_path).dropna(subset=["text", "label"]) if val_path.exists() else None
    test_df = pd.read_csv(test_path).dropna(subset=["text", "label"])

    X_train, y_train = train_df["text"].astype(str), train_df["label"].astype(str)
    X_val, y_val = (val_df["text"].astype(str), val_df["label"].astype(str)) if val_df is not None else (None, None)
    X_test, y_test = test_df["text"].astype(str), test_df["label"].astype(str)

    logger.info(f"Train class distribution: {Counter(y_train)}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val) if y_val is not None else None
    y_test_enc = le.transform(y_test)

    # Save LabelEncoder
    joblib.dump(le, output_dir / "label_encoder.pkl")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(le.classes_)

    # Setup id2label and label2id mapping in config
    id2label = {int(i): str(c) for i, c in enumerate(le.classes_)}
    label2id = {str(c): int(i) for i, c in enumerate(le.classes_)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = NewsDataset(X_train.tolist(), y_train_enc, tokenizer)
    val_dataset = NewsDataset(X_val.tolist(), y_val_enc, tokenizer) if X_val is not None else None
    test_dataset = NewsDataset(X_test.tolist(), y_test_enc, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True if val_dataset else False,
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if val_dataset else None,
    )

    trainer.train()

    # Save model and tokenizer to best_model dir
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    # Evaluate on test set
    preds_output = trainer.predict(test_dataset)
    y_pred = preds_output.predictions.argmax(-1)

    acc = accuracy_score(y_test_enc, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test_enc, y_pred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test_enc, y_pred, average="macro", zero_division=0)

    trained_at = datetime.now(timezone.utc).isoformat()

    metrics_payload = {
        "model_name": "FinBERT (fine-tuned)",
        "base_model": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec_w), 4),
        "recall": round(float(rec_w), 4),
        "macro_f1": round(float(f1_m), 4),
        "weighted_f1": round(float(f1_w), 4),
        "n_test": len(y_test_enc),
        "trained_at": trained_at,
        "id2label": id2label,
        "label2id": label2id,
    }

    # Write metrics.json right next to best_model
    metrics_path = best_model_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved metrics.json to {metrics_path}")

    # Append to models/model_metrics.csv
    metrics_file = root / "models" / "model_metrics.csv"
    row = pd.DataFrame([{
        "Model": "Transformer_FinBERT",
        "Accuracy": metrics_payload["accuracy"],
        "Precision": metrics_payload["precision"],
        "Recall": metrics_payload["recall"],
        "F1": metrics_payload["weighted_f1"],
    }])
    if metrics_file.exists():
        existing = pd.read_csv(metrics_file)
        # remove existing FinBERT row if updating
        existing = existing[existing["Model"] != "Transformer_FinBERT"]
        updated = pd.concat([existing, row], ignore_index=True)
    else:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        updated = row
    updated.to_csv(metrics_file, index=False)

    return metrics_payload


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    print("To start FinBERT training, run this script directly:")
    print("  python -m src.models.transformer")
    print("Expected runtime: ~15-25 minutes on GPU, ~60-90 minutes on CPU.")


if __name__ == "__main__":
    main()
