"""Transformer benchmark for the paper.

Fine-tunes ``bert-base-uncased`` and ``ProsusAI/finbert`` using the
HuggingFace Trainer API.  This file does NOT import anything from
``src.models.transformer``; it is a clean, self-contained implementation
that exactly matches the paper protocol.

Paper protocol (same for both models):
    - Tokenizer : WordPiece, max_length=128, truncation=True
    - Head      : [CLS] → linear + softmax (AutoModelForSequenceClassification)
    - Loss      : CrossEntropy with class weights from train-set frequencies
    - Optimizer : AdamW, lr=2e-5, weight_decay=0.01, warmup_ratio=0.1
    - Training  : batch=16, max 5 epochs, early stopping patience=2 on val macro-F1
    - Best ckpt loaded at end; fp16 only if CUDA; seeds fixed to 42

Output paths
    models/paper_runs/<run_name>/   – saved model + config + tokenizer
    models/paper_runs/<run_name>/metrics.json
    results/paper/predictions/<run_name>.csv
    results/paper/transformer_metrics.csv   (one row appended per run)

CLI::

    # Smoke test (tiny subset, 1 epoch – pipeline verified in ~3-8 min on CPU):
    python -m src.benchmark.transformers_run --models bert --smoke
    python -m src.benchmark.transformers_run --models bert finbert --smoke

    # Full training (~3-6 hours PER MODEL on CPU, run overnight):
    python -m src.benchmark.transformers_run --models bert
    python -m src.benchmark.transformers_run --models bert finbert
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
CLASS_ORDER: List[str] = ["negative", "neutral", "positive"]
SEED: int = 42
MAX_LENGTH: int = 128
TRAIN_BATCH: int = 16
EVAL_BATCH: int = 32
INFER_BATCH: int = 32
LR: float = 2e-5
WEIGHT_DECAY: float = 0.01
WARMUP_RATIO: float = 0.1
NUM_EPOCHS: int = 5
EARLY_STOPPING_PATIENCE: int = 2
N_TIMING_REPEATS: int = 3

# ── smoke-test overrides ───────────────────────────────────────────────────────
SMOKE_N_TRAIN: int = 32
SMOKE_N_VAL: int = 16
SMOKE_N_TEST: int = 16
SMOKE_EPOCHS: int = 1
SMOKE_BATCH: int = 8

# ── model aliases (CLI shorthand → HF model name) ─────────────────────────────
MODEL_ALIASES: Dict[str, str] = {
    "bert": "bert-base-uncased",
    "finbert": "ProsusAI/finbert",
}
# HF model name → canonical run name (used for directory / file names)
RUN_NAMES: Dict[str, str] = {
    "bert-base-uncased": "bert_base",
    "ProsusAI/finbert": "finbert",
}

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _ROOT / "data" / "processed"
_MODEL_DIR = _ROOT / "models" / "paper_runs"
_RESULTS_DIR = _ROOT / "results" / "paper"


# ── dataset wrapper ────────────────────────────────────────────────────────────

class SentimentDataset(TorchDataset):
    """Thin wrapper around a tokenised text list for use with Trainer."""

    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: List[int],
    ) -> None:
        self._encodings = encodings
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self._encodings.items()}
        item["labels"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item


# ── custom trainer with weighted cross-entropy ────────────────────────────────

class WeightedLossTrainer(Trainer):
    """HuggingFace Trainer subclass that uses class-weighted CrossEntropyLoss.

    Args:
        class_weights: 1-D float tensor of length ``num_labels``, ordered to
            match the model's ``id2label`` mapping.  Kept on CPU; moved to the
            correct device inside ``compute_loss``.
    """

    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights.float()

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Override to apply weighted cross-entropy."""
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("inputs missing labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = self._class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_label_maps(
    class_order: List[str],
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Build id2label / label2id dicts from an ordered class list."""
    id2label = {i: c for i, c in enumerate(class_order)}
    label2id = {c: i for i, c in enumerate(class_order)}
    return id2label, label2id


def _compute_class_weights(y_train: List[int], num_classes: int) -> torch.Tensor:
    """Return a float tensor of class weights (balanced, ordered 0..num_classes-1).

    Args:
        y_train: Integer-encoded training labels.
        num_classes: Total number of classes.

    Returns:
        Tensor of shape ``(num_classes,)`` with inverse-frequency weights.
    """
    classes = list(range(num_classes))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=np.array(y_train),
    )
    return torch.tensor(weights, dtype=torch.float)


def _make_compute_metrics(label2id: Dict[str, int]) -> Any:
    """Return a compute_metrics function suitable for HuggingFace Trainer."""

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(labels, preds))
        weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
        prec, rec, _, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {
            "macro_f1": macro_f1,
            "accuracy": acc,
            "weighted_f1": weighted_f1,
            "macro_precision": float(prec),
            "macro_recall": float(rec),
        }

    return compute_metrics


def _dir_size_mb(path: Path) -> float:
    """Return total size of all files under *path* in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def _count_params(model: Any) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _find_best_epoch(log_history: List[Dict[str, Any]]) -> int:
    """Extract the epoch index that achieved the best eval macro_f1."""
    eval_logs = [e for e in log_history if "eval_macro_f1" in e]
    if not eval_logs:
        return -1
    best = max(eval_logs, key=lambda x: x["eval_macro_f1"])
    return int(round(float(best.get("epoch", -1))))


def _time_infer_per_1k(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
) -> float:
    """Return mean inference time in seconds per 1 000 samples.

    Uses batch_size=INFER_BATCH with a warm-up pass before timing.

    Args:
        model: Fitted model in eval mode.
        tokenizer: Corresponding tokenizer.
        texts: List of raw text strings to infer on.
        device: Torch device.

    Returns:
        Mean inference time (seconds) per 1 000 samples across
        ``N_TIMING_REPEATS`` repeats.
    """
    model.eval()
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    n = len(texts)

    def _run_once() -> None:
        with torch.no_grad():
            for i in range(0, n, INFER_BATCH):
                batch = {k: v[i: i + INFER_BATCH].to(device) for k, v in encodings.items()}
                model(**batch)

    _run_once()  # warm-up
    times: List[float] = []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        _run_once()
        elapsed = time.perf_counter() - t0
        times.append(elapsed / n * 1000)
    return float(np.mean(times))


# ── main training function ────────────────────────────────────────────────────

def run_transformer(
    model_name: str,
    run_name: str,
    data_dir: Path = _DATA_DIR,
    output_dir: Path = _MODEL_DIR,
    results_dir: Path = _RESULTS_DIR,
    smoke: bool = False,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Fine-tune a transformer model and evaluate on the test set.

    Args:
        model_name: HuggingFace model identifier, e.g. ``"bert-base-uncased"``.
        run_name: Short name used for output paths, e.g. ``"bert_base"``.
        data_dir: Directory containing ``train.csv``, ``val.csv``, ``test.csv``.
        output_dir: Parent directory for saved model runs
            (model saved to ``output_dir / run_name``).
        results_dir: Root output directory for metrics and prediction CSVs.
        smoke: If True, use a tiny data subset and 1 epoch to verify the
            pipeline quickly.
        seed: Random seed (fixed to 42 for the paper runs).

    Returns:
        Dict containing all computed metrics (same keys as metrics.json).

    Raises:
        FileNotFoundError: If any of the required CSV splits are missing.
        RuntimeError: If the model or tokenizer cannot be loaded.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = torch.cuda.is_available()
    hardware = torch.cuda.get_device_name(0) if fp16 else "CPU"
    logger.info(
        "run_transformer: model=%s run=%s smoke=%s device=%s",
        model_name,
        run_name,
        smoke,
        hardware,
    )

    # ── label maps ────────────────────────────────────────────────────────────
    id2label, label2id = _make_label_maps(CLASS_ORDER)
    num_labels = len(CLASS_ORDER)

    # ── load splits ───────────────────────────────────────────────────────────
    required = {"text", "label"}
    splits: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path).dropna(subset=list(required))
        if missing := required - set(df.columns):
            raise ValueError(f"{split}.csv missing columns: {missing}")
        splits[split] = df

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # ── smoke: subsample ─────────────────────────────────────────────────────
    if smoke:
        logger.info("SMOKE mode: subsampling to %d/%d/%d rows", SMOKE_N_TRAIN, SMOKE_N_VAL, SMOKE_N_TEST)
        train_df = train_df.sample(n=min(SMOKE_N_TRAIN, len(train_df)), random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(SMOKE_N_VAL, len(val_df)), random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(n=min(SMOKE_N_TEST, len(test_df)), random_state=seed).reset_index(drop=True)

    # ── encode labels ─────────────────────────────────────────────────────────
    def encode_labels(series: pd.Series) -> List[int]:
        return [label2id[v] for v in series.astype(str)]

    y_train = encode_labels(train_df["label"])
    y_val = encode_labels(val_df["label"])
    y_test = encode_labels(test_df["label"])

    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    # ── class weights from TRAIN ──────────────────────────────────────────────
    class_weights = _compute_class_weights(y_train, num_labels)
    logger.info("Class weights: %s", dict(zip(CLASS_ORDER, class_weights.tolist())))

    # ── tokenizer ────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,          # dynamic padding via collator
        )

    train_enc = tokenize(train_texts)
    val_enc = tokenize(val_texts)
    test_enc = tokenize(test_texts)

    train_dataset = SentimentDataset(train_enc, y_train)
    val_dataset = SentimentDataset(val_enc, y_val)
    test_dataset = SentimentDataset(test_enc, y_test)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("Loading model: %s  (num_labels=%d)", model_name, num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    param_count = _count_params(model)
    logger.info("Trainable parameters: %s", f"{param_count:,}")

    # ── training args ─────────────────────────────────────────────────────────
    epochs = SMOKE_EPOCHS if smoke else NUM_EPOCHS
    train_batch = SMOKE_BATCH if smoke else TRAIN_BATCH
    eval_batch = SMOKE_BATCH if smoke else EVAL_BATCH

    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
        # Resolve potential API differences between transformers versions
        training_args_kwargs: Dict[str, Any] = dict(
            output_dir=tmp_ckpt_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=eval_batch,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            seed=seed,
            data_seed=seed,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            load_best_model_at_end=True,
            save_total_limit=1,
            fp16=fp16,
            report_to="none",
            logging_steps=10,
            save_strategy="epoch",
            # Use the newer eval_strategy param (transformers ≥ 4.46)
            eval_strategy="epoch",
        )

        try:
            training_args = TrainingArguments(**training_args_kwargs)
        except TypeError:
            # Fallback for older transformers that use evaluation_strategy
            training_args_kwargs.pop("eval_strategy", None)
            training_args_kwargs["evaluation_strategy"] = "epoch"
            training_args = TrainingArguments(**training_args_kwargs)

        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=_make_compute_metrics(label2id),
            callbacks=callbacks,
        )

        logger.info("Starting training…")
        t_train_start = time.perf_counter()
        _ = trainer.train()
        train_seconds = time.perf_counter() - t_train_start
        logger.info("Training finished in %.1f s", train_seconds)

        # ── persist best model ────────────────────────────────────────────────
        trainer.save_model(str(run_output_dir))
        tokenizer.save_pretrained(str(run_output_dir))
        logger.info("Model saved → %s", run_output_dir)

    # ── evaluate on test set ──────────────────────────────────────────────────
    model.eval()
    model.to(device)

    # Collect raw predictions via Trainer.predict
    test_output = trainer.predict(test_dataset)
    logits = test_output.predictions
    y_pred_int = np.argmax(logits, axis=-1).tolist()

    y_pred_labels = [id2label[i] for i in y_pred_int]
    y_test_labels = [id2label[i] for i in y_test]

    acc = float(accuracy_score(y_test_labels, y_pred_labels))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_test_labels, y_pred_labels, average="macro", zero_division=0,
        labels=CLASS_ORDER,
    )
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_test_labels, y_pred_labels, average="weighted", zero_division=0,
        labels=CLASS_ORDER,
    )

    # ── inference timing ──────────────────────────────────────────────────────
    logger.info("Timing inference (batch=%d, ×%d)…", INFER_BATCH, N_TIMING_REPEATS)
    infer_secs_per_1k = _time_infer_per_1k(model, tokenizer, test_texts, device)

    # ── best epoch from log history ───────────────────────────────────────────
    log_history = trainer.state.log_history
    best_epoch = _find_best_epoch(log_history)
    epochs_run = int(round(float(trainer.state.epoch))) if trainer.state.epoch else epochs

    # ── model size on disk ────────────────────────────────────────────────────
    model_size_mb = _dir_size_mb(run_output_dir)
    logger.info("Model size on disk: %.1f MB", model_size_mb)

    # ── save predictions CSV ──────────────────────────────────────────────────
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "index": test_df.index.tolist(),
            "text": test_texts,
            "y_true": y_test_labels,
            "y_pred": y_pred_labels,
        }
    )
    run_key = f"{run_name}_smoke" if smoke else run_name
    pred_path = pred_dir / f"{run_key}.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("Predictions saved → %s", pred_path)

    # ── build metrics dict ────────────────────────────────────────────────────
    trained_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "run_name": run_key,
        "accuracy": round(acc, 6),
        "macro_precision": round(float(prec_m), 6),
        "macro_recall": round(float(rec_m), 6),
        "macro_f1": round(float(f1_m), 6),
        "weighted_f1": round(float(f1_w), 6),
        "n_test": len(test_texts),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "trained_at": trained_at,
        "train_seconds": round(train_seconds, 2),
        "infer_seconds_per_1k": round(infer_secs_per_1k, 6),
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "hardware": hardware,
        "smoke": smoke,
    }

    # ── save metrics JSON to model dir ────────────────────────────────────────
    metrics_path = run_output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved → %s", metrics_path)

    # ── append to transformer_metrics.csv ────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    tf_metrics_path = results_dir / "transformer_metrics.csv"
    row_df = pd.DataFrame([metrics])
    if tf_metrics_path.exists():
        existing = pd.read_csv(tf_metrics_path)
        # Remove any previous row with the same run_name so we don't duplicate
        existing = existing[existing["run_name"] != run_key]
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(tf_metrics_path, index=False)
    logger.info("Appended to %s", tf_metrics_path)

    logger.info(
        "DONE  %s  acc=%.4f  macro_f1=%.4f  train=%.1fs",
        run_key, acc, f1_m, train_seconds,
    )
    return metrics


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entrypoint: ``python -m src.benchmark.transformers_run``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Transformer benchmark for the paper")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_ALIASES.keys()),
        default=["bert", "finbert"],
        help="Which models to run (default: bert finbert)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: tiny subset, 1 epoch — for pipeline verification only",
    )
    args = parser.parse_args()

    for alias in args.models:
        model_name = MODEL_ALIASES[alias]
        run_name = RUN_NAMES[model_name]
        logger.info("=" * 60)
        logger.info("Starting: %s  (run_name=%s)", model_name, run_name)
        logger.info("=" * 60)
        try:
            metrics = run_transformer(
                model_name=model_name,
                run_name=run_name,
                smoke=args.smoke,
            )
        except Exception as exc:
            # Print a clear message rather than a traceback
            logger.error("FAILED: %s — %s: %s", model_name, type(exc).__name__, exc)
            raise SystemExit(1) from exc

        print("\n" + "=" * 60)
        print(f"RESULTS: {metrics['run_name']}")
        print("=" * 60)
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            print(f"  {k:22s}: {metrics[k]:.4f}")
        print(f"  {'epochs_run':22s}: {metrics['epochs_run']}")
        print(f"  {'best_epoch':22s}: {metrics['best_epoch']}")
        print(f"  {'train_seconds':22s}: {metrics['train_seconds']:.1f}")
        print(f"  {'infer_seconds_per_1k':22s}: {metrics['infer_seconds_per_1k']:.6f}")
        print(f"  {'param_count':22s}: {metrics['param_count']:,}")
        print(f"  {'model_size_mb':22s}: {metrics['model_size_mb']} MB")
        print(f"  {'hardware':22s}: {metrics['hardware']}")
        print("=" * 60)

    if args.smoke:
        print("\n[SMOKE DONE] Pipeline verified.")
        print("Full-run commands (run each overnight on CPU, ~3-6 h per model):")
        print("  .venv\\Scripts\\python.exe -m src.benchmark.transformers_run --models bert")
        print("  .venv\\Scripts\\python.exe -m src.benchmark.transformers_run --models finbert")
        print("  # or both at once:")
        print("  .venv\\Scripts\\python.exe -m src.benchmark.transformers_run --models bert finbert")


if __name__ == "__main__":
    main()
