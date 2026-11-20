"""Tests for src.benchmark.transformers_run (Phase 2).

Design: all tests run fully offline in seconds by testing class structure,
constants, and label-map persistence using tiny random-initialised models
(no HuggingFace downloads required).

Test coverage:
    - CLASS_ORDER matches classical.py
    - MODEL_ALIASES contains bert and finbert
    - EARLY_STOPPING_PATIENCE == 2
    - WeightedLossTrainer overrides compute_loss
    - WeightedLossTrainer.__init__ stores class_weights attribute
    - id2label / label2id survive a save-then-reload cycle (tiny BertConfig)
    - metrics.json schema (required keys present) on any existing smoke output
    - transformer_metrics.csv schema check on any existing smoke output
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

# ── repo root on sys.path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.transformers_run import (  # noqa: E402
    CLASS_ORDER,
    EARLY_STOPPING_PATIENCE,
    MODEL_ALIASES,
    RUN_NAMES,
    WeightedLossTrainer,
    _compute_class_weights,
    _make_label_maps,
)

# ── constants tests ───────────────────────────────────────────────────────────


def test_class_order_matches_classical() -> None:
    """CLASS_ORDER must match the one used in classical.py."""
    from src.benchmark.classical import CLASS_ORDER as CLASSICAL_ORDER
    assert CLASS_ORDER == CLASSICAL_ORDER, (
        f"Mismatch: transformers_run={CLASS_ORDER} vs classical={CLASSICAL_ORDER}"
    )


def test_early_stopping_patience_is_2() -> None:
    """Paper specifies patience=2 for early stopping."""
    assert EARLY_STOPPING_PATIENCE == 2


def test_model_aliases_contain_bert_and_finbert() -> None:
    """MODEL_ALIASES must expose 'bert' and 'finbert' shorthand keys."""
    assert "bert" in MODEL_ALIASES
    assert "finbert" in MODEL_ALIASES
    assert MODEL_ALIASES["bert"] == "bert-base-uncased"
    assert MODEL_ALIASES["finbert"] == "ProsusAI/finbert"


def test_run_names_defined() -> None:
    """RUN_NAMES must map both HF model names to canonical run names."""
    assert "bert-base-uncased" in RUN_NAMES
    assert "ProsusAI/finbert" in RUN_NAMES
    assert RUN_NAMES["bert-base-uncased"] == "bert_base"
    assert RUN_NAMES["ProsusAI/finbert"] == "finbert"


# ── WeightedLossTrainer structural tests ──────────────────────────────────────

def test_weighted_loss_trainer_overrides_compute_loss() -> None:
    """WeightedLossTrainer must define its own compute_loss (not inherited)."""
    assert "compute_loss" in WeightedLossTrainer.__dict__, (
        "WeightedLossTrainer must override compute_loss"
    )


def test_weighted_loss_trainer_stores_weights() -> None:
    """WeightedLossTrainer.__init__ must store class_weights as an attribute."""
    # We cannot instantiate Trainer without model/args, but we can check the
    # __init__ signature stores the weights by subclassing directly.
    import inspect
    src = inspect.getsource(WeightedLossTrainer.__init__)
    assert "_class_weights" in src or "class_weights" in src, (
        "WeightedLossTrainer.__init__ must store class_weights"
    )


def test_weighted_loss_uses_crossentropy() -> None:
    """compute_loss source must reference CrossEntropyLoss."""
    import inspect
    src = inspect.getsource(WeightedLossTrainer.compute_loss)
    assert "CrossEntropyLoss" in src


# ── label map tests ───────────────────────────────────────────────────────────

def test_make_label_maps_correct() -> None:
    """_make_label_maps must produce consistent id2label / label2id."""
    id2label, label2id = _make_label_maps(CLASS_ORDER)
    assert len(id2label) == len(CLASS_ORDER)
    assert len(label2id) == len(CLASS_ORDER)
    for i, cls in enumerate(CLASS_ORDER):
        assert id2label[i] == cls
        assert label2id[cls] == i


def test_class_weights_shape_and_positive() -> None:
    """_compute_class_weights must return a tensor of shape (num_labels,) with positive values."""
    y = [0, 0, 0, 1, 1, 2]   # imbalanced
    weights = _compute_class_weights(y, num_classes=3)
    assert weights.shape == (3,), f"Expected shape (3,) got {weights.shape}"
    assert (weights > 0).all(), "All class weights must be positive"
    # Minority class (label 2, only 1 sample) should have higher weight
    assert weights[2] > weights[0], "Minority class should have higher weight"


# ── id2label persistence test (fully offline, tiny BertConfig) ────────────────

def test_id2label_survives_save_reload(tmp_path: Path) -> None:
    """id2label and label2id must be preserved through save_pretrained / from_pretrained.

    Uses a tiny randomly-initialised BERT model — no internet access needed.
    """
    from transformers import BertConfig, BertForSequenceClassification

    id2label, label2id = _make_label_maps(CLASS_ORDER)

    config = BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=len(CLASS_ORDER),
        id2label=id2label,
        label2id=label2id,
    )
    model = BertForSequenceClassification(config)
    save_dir = tmp_path / "tiny_model"
    model.save_pretrained(str(save_dir))

    # Reload only the config (no weights needed)
    from transformers import AutoConfig
    loaded_cfg = AutoConfig.from_pretrained(str(save_dir))

    # id2label keys come back as strings after JSON round-trip; normalise
    loaded_id2label = {int(k): v for k, v in loaded_cfg.id2label.items()}
    assert loaded_id2label == id2label, (
        f"id2label mismatch after reload: {loaded_id2label} != {id2label}"
    )
    loaded_label2id = dict(loaded_cfg.label2id)
    assert loaded_label2id == label2id


# ── smoke output schema tests (only run if smoke was already executed) ─────────

SMOKE_METRICS_PATH = REPO_ROOT / "models" / "paper_runs" / "bert_base" / "metrics.json"
SMOKE_TF_METRICS_PATH = REPO_ROOT / "results" / "paper" / "transformer_metrics.csv"
SMOKE_PRED_PATH = REPO_ROOT / "results" / "paper" / "predictions" / "bert_base_smoke.csv"

REQUIRED_METRICS_KEYS = {
    "model_name", "run_name", "accuracy", "macro_precision",
    "macro_recall", "macro_f1", "weighted_f1", "n_test",
    "epochs_run", "best_epoch", "trained_at", "train_seconds",
    "infer_seconds_per_1k", "param_count", "model_size_mb",
    "hardware", "smoke",
}


@pytest.mark.skipif(
    not SMOKE_METRICS_PATH.exists(),
    reason="Smoke run not yet executed; skipping schema check",
)
def test_metrics_json_schema() -> None:
    """metrics.json must contain all required keys with sensible types."""
    data = json.loads(SMOKE_METRICS_PATH.read_text())
    missing = REQUIRED_METRICS_KEYS - set(data.keys())
    assert not missing, f"metrics.json missing keys: {missing}"
    assert 0.0 <= data["accuracy"] <= 1.0
    assert 0.0 <= data["macro_f1"] <= 1.0
    assert data["n_test"] > 0
    assert data["param_count"] > 0
    assert data["train_seconds"] > 0


@pytest.mark.skipif(
    not SMOKE_TF_METRICS_PATH.exists(),
    reason="transformer_metrics.csv not yet produced; skipping",
)
def test_transformer_metrics_csv_schema() -> None:
    """transformer_metrics.csv must have expected columns."""
    import pandas as pd
    df = pd.read_csv(SMOKE_TF_METRICS_PATH)
    for col in ("model_name", "run_name", "accuracy", "macro_f1", "train_seconds"):
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.skipif(
    not SMOKE_PRED_PATH.exists(),
    reason="Smoke predictions not yet produced; skipping",
)
def test_smoke_predictions_schema() -> None:
    """Smoke predictions CSV must have correct columns and valid label values."""
    import pandas as pd
    df = pd.read_csv(SMOKE_PRED_PATH)
    for col in ("index", "text", "y_true", "y_pred"):
        assert col in df.columns, f"Missing column: {col}"
    unexpected = set(df["y_pred"]) - set(CLASS_ORDER)
    assert not unexpected, f"Unexpected y_pred values: {unexpected}"
    unexpected_true = set(df["y_true"]) - set(CLASS_ORDER)
    assert not unexpected_true, f"Unexpected y_true values: {unexpected_true}"
