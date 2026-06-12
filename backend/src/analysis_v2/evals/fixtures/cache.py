"""Gitignored cache for eval datasets too large to commit.

`python -m src.analysis_v2.evals.fixtures.cache` downloads everything
(needs Kaggle credentials). Tests skip, never fail, when a cached file
is absent, so the hermetic suite stays green offline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "cache"

DATASETS: dict[str, tuple[str, str]] = {
    # key -> (kaggle slug, filename inside the bundle)
    "hillstrom": (
        "bofulee/kevin-hillstrom-minethatdata-e-mailanalytics",
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    ),
    "telco": (
        "blastchar/telco-customer-churn",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ),
}


def cached_path(key: str) -> Path | None:
    """The cached file, or None when it has not been fetched."""
    _, filename = DATASETS[key]
    path = CACHE_DIR / filename
    return path if path.exists() else None


def load(key: str) -> pd.DataFrame:
    path = cached_path(key)
    if path is None:
        raise FileNotFoundError(
            f"eval dataset '{key}' not cached; run "
            "python -m src.analysis_v2.evals.fixtures.cache"
        )
    return pd.read_csv(path)


def fetch_all() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    CACHE_DIR.mkdir(exist_ok=True)
    for key, (slug, filename) in DATASETS.items():
        if (CACHE_DIR / filename).exists():
            print(f"{key}: cached")
            continue
        print(f"{key}: downloading {slug}")
        api.dataset_download_files(slug, path=str(CACHE_DIR), unzip=True)


if __name__ == "__main__":
    fetch_all()
