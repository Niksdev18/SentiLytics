"""Smoke tests — verify the test harness itself works."""

from pathlib import Path


def test_project_root_exists():
    """Ensure the project root directory structure is intact."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "src").is_dir(), "src/ directory missing"
    assert (root / "data").is_dir(), "data/ directory missing"
    assert (root / "requirements.txt").is_file(), "requirements.txt missing"
    assert (root / "config_template.yaml").is_file(), "config_template.yaml missing"
    assert (root / ".env.example").is_file(), ".env.example missing"


def test_data_file_exists():
    """Ensure the Financial PhraseBank dataset is present."""
    root = Path(__file__).resolve().parent.parent
    dataset = root / "data" / "Sentences_AllAgree.txt"
    assert dataset.is_file(), f"Dataset not found at {dataset}"
    # Quick sanity: file should have content
    assert dataset.stat().st_size > 0, "Dataset file is empty"


def test_sample_data_exists():
    """Ensure the sample RSS JSON was preserved."""
    root = Path(__file__).resolve().parent.parent
    sample = root / "data" / "sample" / "20250930_google_rss.json"
    assert sample.is_file(), f"Sample RSS not found at {sample}"


def test_legacy_directory_exists():
    """Ensure superseded files were moved, not deleted."""
    root = Path(__file__).resolve().parent.parent
    legacy = root / "legacy"
    assert legacy.is_dir(), "legacy/ directory missing"
    assert (legacy / "README.md").is_file(), "legacy/README.md missing"
    # At least the streamlit app should be there
    assert (legacy / "streamlit_app.py").is_file(), "legacy/streamlit_app.py missing"
