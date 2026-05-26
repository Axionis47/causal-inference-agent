"""Kaggle URL parsing."""
from __future__ import annotations

import pytest

from src.download.url import ParsedKaggleUrl, parse_kaggle_url


def test_canonical_dataset_url():
    got = parse_kaggle_url("https://www.kaggle.com/datasets/uciml/iris")
    assert got == ParsedKaggleUrl(owner="uciml", slug="iris", version=None)
    assert got.dataset_ref == "uciml/iris"


def test_trailing_slash_is_ignored():
    got = parse_kaggle_url("https://www.kaggle.com/datasets/uciml/iris/")
    assert got.slug == "iris"


def test_no_www_subdomain():
    got = parse_kaggle_url("https://kaggle.com/datasets/uciml/iris")
    assert got.owner == "uciml"


def test_subpath_data_is_stripped():
    got = parse_kaggle_url("https://www.kaggle.com/datasets/uciml/iris/data")
    assert got.dataset_ref == "uciml/iris"
    assert got.version is None


def test_subpath_discussion_is_stripped():
    got = parse_kaggle_url("https://www.kaggle.com/datasets/uciml/iris/discussion")
    assert got.slug == "iris"


def test_versioned_url():
    got = parse_kaggle_url("https://www.kaggle.com/datasets/uciml/iris/versions/3")
    assert got.version == "3"


def test_query_parameters_are_ignored():
    got = parse_kaggle_url("https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv")
    assert got.dataset_ref == "uciml/iris"


def test_whitespace_is_stripped():
    got = parse_kaggle_url("  https://www.kaggle.com/datasets/uciml/iris  ")
    assert got.slug == "iris"


def test_rejects_non_kaggle_url():
    with pytest.raises(ValueError, match="Not a Kaggle URL"):
        parse_kaggle_url("https://example.com/datasets/uciml/iris")


def test_rejects_kaggle_non_dataset_url():
    with pytest.raises(ValueError, match="must be of the form"):
        parse_kaggle_url("https://www.kaggle.com/code/uciml/iris")


def test_rejects_missing_slug():
    with pytest.raises(ValueError, match="must be of the form"):
        parse_kaggle_url("https://www.kaggle.com/datasets/uciml")


def test_rejects_empty_url():
    with pytest.raises(ValueError, match="non-empty"):
        parse_kaggle_url("")


def test_rejects_non_string():
    with pytest.raises(ValueError, match="non-empty"):
        parse_kaggle_url(None)  # type: ignore[arg-type]
