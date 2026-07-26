"""Fetch a dataset from Kaggle and hand back one CSV to analyse.

Deliberately small. It does three things: work out the slug from whatever the
person pasted, download once into a cache, and pick a file. Everything about
what the data *means* is the pipeline's job.

The file choice is the only judgement here, and it is made in the open: with
several CSVs the largest is used and the rest are named in the result, so a
wrong pick is visible rather than silent.
"""
from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path(__file__).parent.parent / "data" / "kaggle"

SLUG = re.compile(r"^[\w.-]+/[\w.-]+$")


class KaggleError(RuntimeError):
    """Anything that stops us getting a usable CSV. The message says what."""


def slug_from(text: str) -> str:
    """Accept a full URL, a bare slug, or something close to either.

    kaggle.com/datasets/owner/name  ->  owner/name
    kaggle.com/owner/name/versions/3 ->  owner/name
    """
    text = text.strip().rstrip("/")
    if SLUG.match(text):
        return text
    match = re.search(r"kaggle\.com/(?:datasets/)?([\w.-]+/[\w.-]+)", text)
    if match:
        return match.group(1)
    raise KaggleError(
        f"cannot read a dataset slug from {text!r}; "
        "expected owner/name or a kaggle.com dataset URL"
    )


@dataclass
class Fetched:
    slug: str
    csv: Path
    all_csvs: list[str] = field(default_factory=list)
    note: str = ""


def fetch(url_or_slug: str) -> Fetched:
    """Download (once) and return the CSV to analyse."""
    slug = slug_from(url_or_slug)
    target = CACHE / slug.replace("/", "__")

    if not target.exists():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:  # pragma: no cover - environment problem
            raise KaggleError("the kaggle package is not installed") from exc

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            raise KaggleError(
                "kaggle credentials rejected; check ~/.kaggle/kaggle.json"
            ) from exc

        target.mkdir(parents=True, exist_ok=True)
        try:
            api.dataset_download_files(slug, path=str(target), quiet=True, unzip=True)
        except Exception as exc:
            # leave no half-downloaded directory behind to be reused as a cache hit
            for leftover in target.glob("*"):
                leftover.unlink(missing_ok=True)
            target.rmdir()
            raise KaggleError(f"could not download {slug}: {exc}") from exc

        # unzip=True usually handles this, but some datasets still land zipped
        for archive in target.glob("*.zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)
            archive.unlink()

    csvs = sorted(target.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        found = [p.name for p in target.rglob("*") if p.is_file()][:8]
        raise KaggleError(
            f"{slug} has no CSV file" + (f"; it contains {found}" if found else "")
        )

    note = ""
    if len(csvs) > 1:
        note = (
            f"{len(csvs)} CSVs in this dataset; using the largest "
            f"({csvs[0].name}). Others: {', '.join(p.name for p in csvs[1:6])}"
        )
    return Fetched(slug=slug, csv=csvs[0], all_csvs=[p.name for p in csvs], note=note)
