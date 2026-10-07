"""Dataset preparation module for Financial PhraseBank.

Parses data/Sentences_AllAgree.txt into data/raw/financial_phrasebank_clean.csv
with columns: sentence, label.
"""

from pathlib import Path
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def parse_financial_phrasebank(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Parse Sentences_AllAgree.txt into a DataFrame and optionally save to CSV.

    Args:
        input_path: Path to Sentences_AllAgree.txt. Defaults to data/Sentences_AllAgree.txt.
        output_path: Path to write clean CSV. If None, does not write to disk.

    Returns:
        pd.DataFrame with columns ['sentence', 'label'].
    """
    if input_path is None:
        input_path = Path(__file__).resolve().parent.parent.parent / "data" / "Sentences_AllAgree.txt"
    else:
        input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Source dataset not found at {input_path}")

    records = []
    with open(input_path, "r", encoding="latin-1") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            # rsplit on the LAST '@'
            parts = line_str.rsplit("@", 1)
            if len(parts) == 2:
                sentence, label = parts[0].strip(), parts[1].strip().lower()
                records.append({"sentence": sentence, "label": label})

    df = pd.DataFrame(records)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved {len(df)} rows to {output_path}")

    return df


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parent.parent.parent
    src_file = root / "data" / "Sentences_AllAgree.txt"
    dst_file = root / "data" / "raw" / "financial_phrasebank_clean.csv"
    df = parse_financial_phrasebank(src_file, dst_file)
    print(f"Parsed {len(df)} rows:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
