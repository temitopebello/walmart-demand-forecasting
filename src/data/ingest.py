"""Weekly ingest from Kaggle, writing raw CSVs into data/raw/."""
import os
from pathlib import Path
import subprocess


RAW_DIR = Path("data/raw")


def fetch_kaggle_dataset(competition: str = "walmart-recruiting-store-sales-forecasting") -> None:
    """Download the Walmart Kaggle competition dataset.

    Requires KAGGLE_USERNAME and KAGGLE_KEY either in env or ~/.kaggle/kaggle.json.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(RAW_DIR)],
        check=True,
    )


if __name__ == "__main__":
    fetch_kaggle_dataset()
    print("Ingest complete.")
