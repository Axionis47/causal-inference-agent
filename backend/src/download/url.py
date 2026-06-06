"""Kaggle dataset URL parsing.

Kaggle URLs come in many shapes:
    https://www.kaggle.com/datasets/owner/slug
    https://www.kaggle.com/datasets/owner/slug/
    https://www.kaggle.com/datasets/owner/slug/data
    https://www.kaggle.com/datasets/owner/slug/versions/3
    https://www.kaggle.com/datasets/owner/slug?select=file.csv
    https://kaggle.com/datasets/owner/slug

All of them resolve to (owner, slug, version|None). Anything else
raises ValueError with a clear message; we do not silently accept
non-Kaggle URLs.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ParsedKaggleUrl:
    owner: str
    slug: str
    version: str | None  # None means "latest"

    @property
    def dataset_ref(self) -> str:
        return f"{self.owner}/{self.slug}"


def parse_kaggle_url(url: str) -> ParsedKaggleUrl:
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if host not in {"www.kaggle.com", "kaggle.com"}:
        raise ValueError(f"Not a Kaggle URL: {url}")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(
            f"Kaggle dataset URL must be of the form "
            f"https://www.kaggle.com/datasets/<owner>/<slug>, got: {url}"
        )

    owner, slug = parts[1], parts[2]
    if not owner or not slug:
        raise ValueError(f"Owner and slug are required in Kaggle URL: {url}")

    version: str | None = None
    if len(parts) >= 5 and parts[3] == "versions":
        version = parts[4]

    return ParsedKaggleUrl(owner=owner, slug=slug, version=version)
