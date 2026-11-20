"""Classical ML benchmark for the paper.

Trains Logistic Regression, Linear SVM, Multinomial Naive Bayes, and Random
Forest on TF-IDF features derived from the ``text_classic`` column.

Hyperparameters are chosen by evaluating each candidate on the VALIDATION set
(macro-F1), never on the test set.  Class imbalance is handled with
``class_weight='balanced'`` (LR, SVM, RF) or via sample weights derived from
class frequencies (MNB, which does not support class_weight natively).
Oversampling (e.g. synthetic minority methods) is NOT used; the paper uses
only class-weighting / sample-weighting.

All seeds are fixed to 42.  Every number written to disk comes from an actual
run of this script.

Output files (all under ``results/paper/``):
    predictions/<model_key>.csv          – index, text, y_true, y_pred
    classical_best_params.json           – chosen hyper-parameters per model
    classical_metrics.csv                – accuracy / precision / recall /
                                           macro_f1 / weighted_f1 /
                                           train_seconds / infer_seconds_per_1k /
                                           model_size_kb

CLI::

    python -m src.benchmark.classical
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
CLASS_ORDER: List[str] = ["negative", "neutral", "positive"]
SEED: int = 42
TFIDF_MAX_FEATURES: int = 7000
TFIDF_NGRAM: Tuple[int, int] = (1, 3)   # matches baseline.py
N_TIMING_REPEATS: int = 3
INFER_BATCH_SIZE: int = 1000

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _ROOT / "data" / "processed"
_RESULTS_DIR = _ROOT / "results" / "paper"
_PRED_DIR = _RESULTS_DIR / "predictions"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_splits(
    data_dir: Path = _DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train / val / test CSVs and validate required columns.

    Args:
        data_dir: Directory containing ``train.csv``, ``val.csv``, ``test.csv``.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        FileNotFoundError: If any split file is missing.
        ValueError: If required columns are absent.
    """
    required_cols = {"text_classic", "label"}
    dfs: List[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_csv(path).dropna(subset=list(required_cols))
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{split}.csv is missing columns: {missing}")
        dfs.append(df)
    return tuple(dfs)  # type: ignore[return-value]


def _model_size_kb(model: Any) -> float:
    """Serialize *model* to a temp file and return its size in KB."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as tmp:
        tmp_path = Path(tmp.name)
    joblib.dump(model, tmp_path)
    size_kb = tmp_path.stat().st_size / 1024.0
    tmp_path.unlink(missing_ok=True)
    return round(size_kb, 2)


def _time_fit(model: Any, X: Any, y: Any, sample_weight: Any = None) -> float:
    """Fit *model* ``N_TIMING_REPEATS`` times and return mean seconds.

    Args:
        model: A scikit-learn estimator with a ``fit`` method.
        X: Feature matrix.
        y: Label array.
        sample_weight: Optional sample weight array (passed when supported).

    Returns:
        Mean fit time in seconds across repeats.
    """
    times: List[float] = []
    for _ in range(N_TIMING_REPEATS):
        m_clone = pickle.loads(pickle.dumps(model))   # fast deep-copy via pickle
        t0 = time.perf_counter()
        if sample_weight is not None:
            m_clone.fit(X, y, sample_weight=sample_weight)
        else:
            m_clone.fit(X, y)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def _time_infer_per_1k(model: Any, X: Any) -> float:
    """Predict on a 1 000-sample batch and return mean seconds per 1 k samples.

    A warm-up call is made before timing.

    Args:
        model: Fitted estimator with a ``predict`` method.
        X: Feature matrix (at least ``INFER_BATCH_SIZE`` rows, or the full set
           if smaller).

    Returns:
        Mean inference time in seconds per 1 000 samples.
    """
    n = min(INFER_BATCH_SIZE, X.shape[0])
    X_batch = X[:n]
    model.predict(X_batch)   # warm-up
    times: List[float] = []
    for _ in range(N_TIMING_REPEATS):
        t0 = time.perf_counter()
        model.predict(X_batch)
        elapsed = time.perf_counter() - t0
        times.append(elapsed / n * 1000)
    return float(np.mean(times))


def _val_macro_f1(
    model: Any,
    X_val: Any,
    y_val: np.ndarray,
) -> float:
    """Return macro-F1 of *model* on the validation features."""
    y_pred = model.predict(X_val)
    return float(f1_score(y_val, y_pred, average="macro", zero_division=0))


def _grid_search_val(
    candidates: List[Any],
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    sample_weight_train: np.ndarray | None = None,
) -> Tuple[Any, float]:
    """Evaluate each candidate estimator on the validation set and return best.

    Candidates are fitted on train and evaluated on val by macro-F1.

    Args:
        candidates: List of un-fitted scikit-learn estimators to try.
        X_train: Training feature matrix.
        y_train: Training labels (integer-encoded).
        X_val: Validation feature matrix.
        y_val: Validation labels (integer-encoded).
        sample_weight_train: Optional per-sample weights for training
            (passed to ``fit`` only for models that accept it).

    Returns:
        Tuple of (best_estimator, best_val_macro_f1).  The best estimator is
        already fitted on the training set.
    """
    best_score = -1.0
    best_model: Any = None
    for candidate in candidates:
        try:
            if sample_weight_train is not None:
                candidate.fit(X_train, y_train, sample_weight=sample_weight_train)
            else:
                candidate.fit(X_train, y_train)
        except TypeError:
            # Fallback: model doesn't accept sample_weight
            candidate.fit(X_train, y_train)
        score = _val_macro_f1(candidate, X_val, y_val)
        logger.debug("  candidate=%s  val_macro_f1=%.4f", candidate, score)
        if score > best_score:
            best_score = score
            best_model = candidate
    return best_model, best_score


# ── per-model training routines ───────────────────────────────────────────────

def _build_lr_candidates() -> List[LogisticRegression]:
    """Return a list of LR candidates with varying C."""
    return [
        LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
            # multi_class removed in sklearn 1.5; lbfgs always uses multinomial
        )
        for c in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ]


def _build_svm_candidates() -> List[LinearSVC]:
    """Return a list of LinearSVC candidates with varying C."""
    return [
        LinearSVC(
            C=c,
            class_weight="balanced",
            random_state=SEED,
            max_iter=3000,
        )
        for c in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ]


def _build_mnb_candidates() -> List[MultinomialNB]:
    """Return a list of MNB candidates with varying alpha."""
    return [MultinomialNB(alpha=a) for a in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]]


def _build_rf_candidates() -> List[RandomForestClassifier]:
    """Return RF candidates with varying max_depth; n_estimators fixed at 300."""
    return [
        RandomForestClassifier(
            n_estimators=300,
            max_depth=d,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        for d in [None, 10, 20, 30, 50]
    ]


# ── main benchmark function ───────────────────────────────────────────────────

def run_classical_benchmark(
    data_dir: Path = _DATA_DIR,
    results_dir: Path = _RESULTS_DIR,
) -> pd.DataFrame:
    """Run the full classical ML benchmark and write all output files.

    Steps:
        1. Load splits, encode labels with the fixed CLASS_ORDER.
        2. Fit TF-IDF on TRAIN only (ngram (1,3), max_features 7000).
        3. For each model family, grid-search on VALIDATION macro-F1.
        4. Refit the winning config on TRAIN, predict on TEST.
        5. Time training and inference; measure model size on disk.
        6. Save predictions, best params JSON, and metrics CSV.

    Args:
        data_dir: Directory with train/val/test CSVs.
        results_dir: Root output directory (``results/paper/``).

    Returns:
        DataFrame with one row per model and all computed metrics.

    Note:
        MNB does not support ``class_weight`` natively.  Sample weights
        proportional to inverse class frequency are passed to ``fit`` instead.
        This is noted in the saved params JSON (``imbalance_handling`` key).
    """
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data splits from %s", data_dir)
    train_df, val_df, test_df = _load_splits(data_dir)

    # ── label encoding with fixed class order ─────────────────────────────────
    label2idx: Dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}
    idx2label: Dict[int, str] = {i: c for c, i in label2idx.items()}

    def encode(series: pd.Series) -> np.ndarray:
        return np.array([label2idx[v] for v in series.astype(str)])

    y_train = encode(train_df["label"])
    y_val = encode(val_df["label"])
    y_test = encode(test_df["label"])

    X_train_raw = train_df["text_classic"].astype(str).tolist()
    X_val_raw = val_df["text_classic"].astype(str).tolist()
    X_test_raw = test_df["text_classic"].astype(str).tolist()

    # ── TF-IDF fitted on TRAIN only ──────────────────────────────────────────
    logger.info(
        "Fitting TF-IDF (ngram=%s, max_features=%d) on train…",
        TFIDF_NGRAM,
        TFIDF_MAX_FEATURES,
    )
    tfidf = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )
    X_train_tf = tfidf.fit_transform(X_train_raw)
    X_val_tf = tfidf.transform(X_val_raw)
    X_test_tf = tfidf.transform(X_test_raw)

    # ── sample weights for MNB (inverse class frequency) ──────────────────────
    sw_train = compute_sample_weight("balanced", y_train)

    # ── model definitions ──────────────────────────────────────────────────────
    model_specs: List[Dict[str, Any]] = [
        {
            "key": "naive_bayes",
            "display": "Naive Bayes (TF-IDF)",
            "candidates_fn": _build_mnb_candidates,
            "use_sample_weight": True,       # MNB: no class_weight; use sw
            "imbalance_handling": "sample_weight (inverse class freq)",
        },
        {
            "key": "logistic_regression",
            "display": "Logistic Regression",
            "candidates_fn": _build_lr_candidates,
            "use_sample_weight": False,
            "imbalance_handling": "class_weight='balanced'",
        },
        {
            "key": "svm",
            "display": "Linear SVM",
            "candidates_fn": _build_svm_candidates,
            "use_sample_weight": False,
            "imbalance_handling": "class_weight='balanced'",
        },
        {
            "key": "random_forest",
            "display": "Random Forest",
            "candidates_fn": _build_rf_candidates,
            "use_sample_weight": False,
            "imbalance_handling": "class_weight='balanced'",
        },
    ]

    metrics_rows: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] = {}

    for spec in model_specs:
        key: str = spec["key"]
        display: str = spec["display"]
        logger.info("=== %s: grid-searching on validation set ===", display)

        candidates = spec["candidates_fn"]()
        sw = sw_train if spec["use_sample_weight"] else None

        best_model, best_val_f1 = _grid_search_val(
            candidates, X_train_tf, y_train, X_val_tf, y_val,
            sample_weight_train=sw,
        )
        logger.info(
            "%s: best val macro-F1=%.4f | best config=%s",
            display,
            best_val_f1,
            best_model.get_params(),
        )

        # ── refit best config on train ────────────────────────────────────────
        if sw is not None:
            best_model.fit(X_train_tf, y_train, sample_weight=sw)
        else:
            best_model.fit(X_train_tf, y_train)

        # ── predict on test ───────────────────────────────────────────────────
        y_pred = best_model.predict(X_test_tf)
        y_pred_labels = [idx2label[i] for i in y_pred]
        y_test_labels = [idx2label[i] for i in y_test]

        # ── compute metrics ───────────────────────────────────────────────────
        acc = float(accuracy_score(y_test_labels, y_pred_labels))
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_test_labels,
            y_pred_labels,
            average="macro",
            zero_division=0,
            labels=CLASS_ORDER,
        )
        _, _, f1_w, _ = precision_recall_fscore_support(
            y_test_labels,
            y_pred_labels,
            average="weighted",
            zero_division=0,
            labels=CLASS_ORDER,
        )

        # ── timing ────────────────────────────────────────────────────────────
        logger.info("%s: timing train (×%d)…", display, N_TIMING_REPEATS)
        train_secs = _time_fit(best_model, X_train_tf, y_train, sample_weight=sw)

        logger.info("%s: timing inference (×%d)…", display, N_TIMING_REPEATS)
        infer_secs_per_1k = _time_infer_per_1k(best_model, X_test_tf)

        # ── model size ────────────────────────────────────────────────────────
        size_kb = _model_size_kb(best_model)

        # ── save predictions ──────────────────────────────────────────────────
        pred_df = pd.DataFrame(
            {
                "index": test_df.index.tolist(),
                "text": test_df["text_classic"].astype(str).tolist(),
                "y_true": y_test_labels,
                "y_pred": y_pred_labels,
            }
        )
        pred_path = pred_dir / f"{key}.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info("Predictions saved → %s", pred_path)

        # ── record best params ────────────────────────────────────────────────
        params = best_model.get_params()
        params["imbalance_handling"] = spec["imbalance_handling"]
        params["best_val_macro_f1"] = round(best_val_f1, 6)
        best_params[key] = params

        # ── record metrics row ────────────────────────────────────────────────
        metrics_rows.append(
            {
                "model": display,
                "model_key": key,
                "accuracy": round(acc, 6),
                "macro_precision": round(float(prec_m), 6),
                "macro_recall": round(float(rec_m), 6),
                "macro_f1": round(float(f1_m), 6),
                "weighted_f1": round(float(f1_w), 6),
                "train_seconds": round(train_secs, 4),
                "infer_seconds_per_1k": round(infer_secs_per_1k, 6),
                "model_size_kb": size_kb,
            }
        )

    # ── write outputs ──────────────────────────────────────────────────────────
    params_path = results_dir / "classical_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2, default=str))
    logger.info("Best params saved → %s", params_path)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = results_dir / "classical_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved → %s", metrics_path)

    return metrics_df


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entrypoint: ``python -m src.benchmark.classical``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting classical benchmark")
    df = run_classical_benchmark()
    print("\n" + "=" * 70)
    print("CLASSICAL BENCHMARK — TEST SET RESULTS")
    print("=" * 70)
    cols = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    print(df[cols].to_string(index=False))
    print("\nTiming & size:")
    t_cols = ["model", "train_seconds", "infer_seconds_per_1k", "model_size_kb"]
    print(df[t_cols].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
