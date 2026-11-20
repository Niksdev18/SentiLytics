"""Tests for src.benchmark.classical (Phase 1).

Uses a small in-memory fixture so the tests run without touching the real
data/processed/ files and complete in a few seconds.

Test coverage:
    - Output files are created (predictions, params JSON, metrics CSV).
    - Metrics are in [0, 1].
    - CLASS_ORDER is respected in prediction CSVs.
    - No SMOTE import exists anywhere in classical.py.
    - Two identical runs (seed 42) produce byte-identical prediction CSVs.
    - Timing columns are positive numbers.
    - Model size column is positive.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── locate repo root & add to sys.path ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.classical import CLASS_ORDER, run_classical_benchmark  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────

LABELS = CLASS_ORDER   # ["negative", "neutral", "positive"]
RNG = np.random.default_rng(0)


def _make_split(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a tiny synthetic split with text_classic and label columns."""
    texts = [" ".join(f"word{rng.integers(0, 50)}" for _ in range(10)) for _ in range(n)]
    labels = [LABELS[i % len(LABELS)] for i in range(n)]
    rng.shuffle(labels)  # type: ignore[arg-type]
    return pd.DataFrame({"text": texts, "text_classic": texts, "label": labels})


@pytest.fixture()
def fixture_data_dir(tmp_path: Path) -> Path:
    """Write tiny train/val/test CSVs and return the directory path."""
    rng = np.random.default_rng(0)
    train_df = _make_split(90, rng)
    val_df = _make_split(30, rng)
    test_df = _make_split(30, rng)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


@pytest.fixture()
def fixture_results_dir(tmp_path: Path) -> Path:
    """Return an empty temp directory as the results root."""
    results = tmp_path / "results" / "paper"
    results.mkdir(parents=True)
    return results


# ── test: output files are created ───────────────────────────────────────────

def test_output_files_exist(fixture_data_dir: Path, fixture_results_dir: Path) -> None:
    """All expected output files must be created after a successful run."""
    run_classical_benchmark(
        data_dir=fixture_data_dir,
        results_dir=fixture_results_dir,
    )
    # Per-model prediction CSVs
    pred_dir = fixture_results_dir / "predictions"
    for key in ("naive_bayes", "logistic_regression", "svm", "random_forest"):
        assert (pred_dir / f"{key}.csv").exists(), f"Missing predictions/{key}.csv"

    # Aggregate files
    assert (fixture_results_dir / "classical_best_params.json").exists()
    assert (fixture_results_dir / "classical_metrics.csv").exists()


# ── test: metrics are in [0, 1] ──────────────────────────────────────────────

def test_metrics_in_range(fixture_data_dir: Path, fixture_results_dir: Path) -> None:
    """Accuracy, precision, recall, macro_f1, weighted_f1 must all be in [0, 1]."""
    df = run_classical_benchmark(
        data_dir=fixture_data_dir,
        results_dir=fixture_results_dir,
    )
    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    for col in metric_cols:
        for val in df[col]:
            assert 0.0 <= val <= 1.0, f"{col}={val} is out of [0, 1]"


# ── test: class order is fixed ────────────────────────────────────────────────

def test_class_order_fixed(fixture_data_dir: Path, fixture_results_dir: Path) -> None:
    """y_true and y_pred in every prediction CSV must only contain CLASS_ORDER values."""
    run_classical_benchmark(
        data_dir=fixture_data_dir,
        results_dir=fixture_results_dir,
    )
    pred_dir = fixture_results_dir / "predictions"
    for key in ("naive_bayes", "logistic_regression", "svm", "random_forest"):
        pred_df = pd.read_csv(pred_dir / f"{key}.csv")
        unexpected_true = set(pred_df["y_true"]) - set(CLASS_ORDER)
        unexpected_pred = set(pred_df["y_pred"]) - set(CLASS_ORDER)
        assert not unexpected_true, f"{key}: unexpected y_true values {unexpected_true}"
        assert not unexpected_pred, f"{key}: unexpected y_pred values {unexpected_pred}"


# ── test: no SMOTE import ─────────────────────────────────────────────────────

def test_no_smote_import() -> None:
    """classical.py must not import imblearn or instantiate SMOTE."""
    classical_path = REPO_ROOT / "src" / "benchmark" / "classical.py"
    source = classical_path.read_text(encoding="utf-8")

    # imblearn must not be imported at all
    assert "imblearn" not in source, "classical.py must not import from imblearn"

    # Parse AST: check no import of smote/imblearn, no SMOTE() call
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = ""
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            assert "smote" not in module.lower(), f"SMOTE import found: {module}"
            assert "imblearn" not in module.lower(), f"imblearn import found: {module}"
        # Check no direct SMOTE() call
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name != "SMOTE", "classical.py must not call SMOTE()"


# ── test: seed determinism ────────────────────────────────────────────────────

def test_seed_determinism(fixture_data_dir: Path, tmp_path: Path) -> None:
    """Two consecutive runs must produce byte-identical prediction CSVs."""
    results1 = tmp_path / "run1"
    results2 = tmp_path / "run2"
    results1.mkdir(parents=True)
    results2.mkdir(parents=True)

    run_classical_benchmark(data_dir=fixture_data_dir, results_dir=results1)
    run_classical_benchmark(data_dir=fixture_data_dir, results_dir=results2)

    for key in ("naive_bayes", "logistic_regression", "svm", "random_forest"):
        csv1 = (results1 / "predictions" / f"{key}.csv").read_bytes()
        csv2 = (results2 / "predictions" / f"{key}.csv").read_bytes()
        assert csv1 == csv2, f"Non-deterministic predictions for {key}"


# ── test: timing columns are positive ────────────────────────────────────────

def test_timing_positive(fixture_data_dir: Path, fixture_results_dir: Path) -> None:
    """train_seconds and infer_seconds_per_1k must be positive for every model."""
    df = run_classical_benchmark(
        data_dir=fixture_data_dir,
        results_dir=fixture_results_dir,
    )
    for _, row in df.iterrows():
        assert row["train_seconds"] > 0, f"Non-positive train_seconds for {row['model']}"
        assert row["infer_seconds_per_1k"] > 0, f"Non-positive infer_seconds_per_1k for {row['model']}"


# ── test: model size is positive ──────────────────────────────────────────────

def test_model_size_positive(fixture_data_dir: Path, fixture_results_dir: Path) -> None:
    """model_size_kb must be positive for every model."""
    df = run_classical_benchmark(
        data_dir=fixture_data_dir,
        results_dir=fixture_results_dir,
    )
    for _, row in df.iterrows():
        assert row["model_size_kb"] > 0, f"Non-positive model_size_kb for {row['model']}"


# ── test: best_params JSON is valid and contains expected keys ────────────────

def test_best_params_json(fixture_data_dir: Path, fixture_results_dir: Path) -> None:
    """classical_best_params.json must be valid JSON with one entry per model."""
    run_classical_benchmark(data_dir=fixture_data_dir, results_dir=fixture_results_dir)
    params_path = fixture_results_dir / "classical_best_params.json"
    params = json.loads(params_path.read_text())
    expected_keys = {"naive_bayes", "logistic_regression", "svm", "random_forest"}
    assert set(params.keys()) == expected_keys
    for key in expected_keys:
        assert "imbalance_handling" in params[key], f"Missing imbalance_handling for {key}"
        assert "best_val_macro_f1" in params[key], f"Missing best_val_macro_f1 for {key}"
