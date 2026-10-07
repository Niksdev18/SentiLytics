"""Dataset splitting module.

Stratifies dataset 70/15/15 (train/val/test) with seed 42.
Saves columns: text (lightly cleaned), text_classic (classically cleaned), label.
"""

from pathlib import Path
from typing import Tuple, Optional
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.prepare_dataset import parse_financial_phrasebank
from src.preprocessing.preprocess import light_clean, classic_clean

logger = logging.getLogger(__name__)


def prepare_and_split_data(
    input_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess and split financial sentiment dataset into train, val, and test.

    Splits stratified 70% train, 15% val, 15% test.
    Columns in each resulting dataframe: 'text', 'text_classic', 'label'.

    Args:
        input_csv: Path to raw cleaned CSV (or Sentences_AllAgree.txt if CSV missing).
        output_dir: Output directory to save train.csv, val.csv, test.csv.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    root = Path(__file__).resolve().parent.parent.parent

    if input_csv is None:
        raw_csv = root / "data" / "raw" / "financial_phrasebank_clean.csv"
        if not raw_csv.exists():
            txt_path = root / "data" / "Sentences_AllAgree.txt"
            df = parse_financial_phrasebank(txt_path, raw_csv)
        else:
            df = pd.read_csv(raw_csv)
    else:
        df = pd.read_csv(input_csv)

    # Ensure required columns exist
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'sentence' and 'label' columns")

    # Generate both cleaning tiers
    logger.info("Applying light cleaning for transformer models...")
    df["text"] = df["sentence"].astype(str).apply(light_clean)

    logger.info("Applying classic cleaning for TF-IDF baseline models...")
    df["text_classic"] = df["sentence"].astype(str).apply(classic_clean)

    # Drop any nulls
    df = df.dropna(subset=["text", "text_classic", "label"])

    # First split: Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        df[["text", "text_classic", "label"]],
        test_size=0.30,
        random_state=seed,
        stratify=df["label"],
    )

    # Second split: Validation (15%) vs Test (15%) from Temp (30%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out / "train.csv", index=False)
        val_df.to_csv(out / "val.csv", index=False)
        test_df.to_csv(out / "test.csv", index=False)
        logger.info(f"Saved splits to {out}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return train_df, val_df, test_df


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "data" / "processed"
    train_df, val_df, test_df = prepare_and_split_data(output_dir=out_dir)
    print(f"Total rows: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"Train size: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
    print(f"Val size:   {len(val_df)} ({val_df['label'].value_counts().to_dict()})")
    print(f"Test size:  {len(test_df)} ({test_df['label'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
