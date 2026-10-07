"""Preprocessing functions for text sentiment analysis.

Provides:
- light_clean: Strips HTML, URLs, and excess whitespace while preserving
  case, stopwords, and punctuation. Used for Transformer/BERT models.
- classic_clean: Cleans, tokenizes, and lemmatizes while preserving
  direction/negation words (no, not, nor, up, down, over, under, above, below, off).
  Used for classical/TF-IDF models.
"""

from pathlib import Path
from typing import Optional, Set
import logging
import re

logger = logging.getLogger(__name__)

DIRECTION_NEGATION_WORDS: Set[str] = {
    "no",
    "not",
    "nor",
    "up",
    "down",
    "over",
    "under",
    "above",
    "below",
    "off",
}

_stop_words_cache: Optional[Set[str]] = None
_lemmatizer_cache = None


def _get_stopwords() -> Set[str]:
    """Lazy-load NLTK stopwords excluding direction and negation words."""
    global _stop_words_cache
    if _stop_words_cache is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            base_stopwords = set(stopwords.words("english"))
        except (LookupError, AttributeError):
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            base_stopwords = set(stopwords.words("english"))

        # Preserve key directional and negation signals
        _stop_words_cache = base_stopwords - DIRECTION_NEGATION_WORDS
    return _stop_words_cache


def _get_lemmatizer():
    """Lazy-load NLTK WordNetLemmatizer."""
    global _lemmatizer_cache
    if _lemmatizer_cache is None:
        import nltk
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("testing")
            _lemmatizer_cache = lemmatizer
        except (LookupError, AttributeError):
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.stem import WordNetLemmatizer
            _lemmatizer_cache = WordNetLemmatizer()
    return _lemmatizer_cache


def light_clean(text: str) -> str:
    """Light cleaning for Transformer models (e.g., FinBERT).

    Strips HTML tags, URLs, and collapses whitespace.
    Preserves case, punctuation, and all stopwords.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Collapse multiple whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def classic_clean(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Cleaning for classical models (TF-IDF, Bag-of-Words).

    Lowercases, removes URLs/HTML, normalizes currency, removes non-essential
    stopwords while strictly keeping negation and direction words.

    Args:
        text: Raw text string.
        remove_stopwords: If True, remove non-directional stopwords.
        lemmatize: If True, lemmatize tokens.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Normalize financial expressions
    text = text.replace("us$", "usd").replace("$", " usd ")

    # 5. Remove special characters (keep alphanumeric, $, %, and space)
    text = re.sub(r"[^a-zA-Z0-9$% ]", " ", text)

    # 6. Tokenize
    tokens = text.split()

    # 7. Remove stopwords (while keeping DIRECTION_NEGATION_WORDS)
    if remove_stopwords:
        stops = _get_stopwords()
        tokens = [w for w in tokens if w not in stops]

    # 8. Lemmatization
    if lemmatize:
        lem = _get_lemmatizer()
        tokens = [lem.lemmatize(w) for w in tokens]

    return " ".join(tokens)


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Backward compatibility alias for classic_clean."""
    return classic_clean(text, remove_stopwords=remove_stopwords, lemmatize=lemmatize)


def main() -> None:
    """CLI / batch execution entrypoint."""
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parent.parent.parent
    raw_path = root / "data" / "raw" / "financial_phrasebank_clean.csv"
    if not raw_path.exists():
        print(f"File not found: {raw_path}. Run prepare_dataset.py first.")
        return

    df = pd.read_csv(raw_path)
    df["light_text"] = df["sentence"].apply(light_clean)
    df["classic_text"] = df["sentence"].apply(classic_clean)
    out_path = root / "data" / "processed" / "financial_news_cleaned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Processed {len(df)} rows and saved to {out_path}")
    print(df[["sentence", "light_text", "classic_text"]].head(3))


if __name__ == "__main__":
    main()
