"""Tests for Phase 1: Training-side correctness."""

from collections import Counter
from pathlib import Path
import pytest

from src.preprocessing.prepare_dataset import parse_financial_phrasebank
from src.preprocessing.preprocess import light_clean, classic_clean
from src.preprocessing.split_data import prepare_and_split_data


def test_prepare_dataset_row_counts_and_distribution():
    """Verify parser returns 2264 rows: 1391 neutral, 570 positive, 303 negative."""
    root = Path(__file__).resolve().parent.parent
    src_file = root / "data" / "Sentences_AllAgree.txt"

    df = parse_financial_phrasebank(src_file)
    assert len(df) == 2264, f"Expected 2264 rows, got {len(df)}"

    counts = df["label"].value_counts().to_dict()
    assert counts.get("neutral") == 1391, f"Expected 1391 neutral, got {counts.get('neutral')}"
    assert counts.get("positive") == 570, f"Expected 570 positive, got {counts.get('positive')}"
    assert counts.get("negative") == 303, f"Expected 303 negative, got {counts.get('negative')}"


def test_light_clean_keeps_direction_and_negation():
    """Verify light_clean preserves 'not', 'down', case, and punctuation while stripping HTML/URLs."""
    raw = "<p>Operating Profit did not go down and was NOT down! Check https://example.com/report</p>"
    cleaned = light_clean(raw)

    # Must preserve case ("Operating Profit", "NOT")
    assert "Operating Profit" in cleaned
    assert "NOT" in cleaned
    # Must preserve "not" and "down"
    assert "not" in cleaned
    assert "down" in cleaned
    # Must remove HTML and URL
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "https://" not in cleaned
    assert cleaned == "Operating Profit did not go down and was NOT down! Check"


def test_classic_clean_preserves_direction_and_negation():
    """Verify classic_clean keeps directional/negation words while stripping general stopwords."""
    raw = "The shares did not go down or up, and remained under pressure below expectations."
    cleaned = classic_clean(raw)

    words = cleaned.split()
    # Preserved direction/negation
    for expected in ["not", "down", "up", "under", "below"]:
        assert expected in words, f"Expected '{expected}' to be preserved in classic_clean output"

    # General stopwords removed (the, and, or)
    for removed in ["the", "and", "or"]:
        assert removed not in words, f"Expected '{removed}' to be stripped by classic_clean"


def test_split_preserves_ratios_and_columns(tmp_path):
    """Verify 70/15/15 stratified split and correct output columns without deduplication."""
    root = Path(__file__).resolve().parent.parent
    src_file = root / "data" / "Sentences_AllAgree.txt"
    raw_csv = tmp_path / "raw.csv"

    # Generate raw CSV
    parse_financial_phrasebank(src_file, raw_csv)

    # Perform stratified split (no deduplication, exactly 2264 rows)
    out_dir = tmp_path / "splits"
    train_df, val_df, test_df = prepare_and_split_data(raw_csv, out_dir, seed=42)

    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == 2264, f"Expected total rows to be 2264 without deduplication, got {total_rows}"

    # Check 70/15/15 ratios (within ±2 rows due to integer rounding)
    assert abs(len(train_df) - 0.70 * total_rows) <= 2
    assert abs(len(val_df) - 0.15 * total_rows) <= 2
    assert abs(len(test_df) - 0.15 * total_rows) <= 2

    # Verify columns
    for df in [train_df, val_df, test_df]:
        assert list(df.columns) == ["text", "text_classic", "label"]

    # Verify stratified class proportions: neutral ~61.4%, positive ~25.2%, negative ~13.4%
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        props = df["label"].value_counts(normalize=True)
        assert abs(props["neutral"] - (1391 / 2264)) < 0.02, f"Neutral ratio mismatch in {name}"
        assert abs(props["positive"] - (570 / 2264)) < 0.02, f"Positive ratio mismatch in {name}"
        assert abs(props["negative"] - (303 / 2264)) < 0.02, f"Negative ratio mismatch in {name}"
