"""Profile store: encrypted Kaggle credential persistence.

These tests pin two properties that matter most:
1. The key is NEVER written in plaintext on disk.
2. A second process with the same encryption_key can decrypt; without
   it, decryption raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from cryptography.fernet import Fernet

from src.download import profile_store


@pytest.fixture
def enc_key() -> str:
    return Fernet.generate_key().decode()


def test_set_kaggle_key_persists_profile_and_blob(tmp_path, enc_key):
    profile = profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret-kaggle-key-value",
        root=tmp_path,
        encryption_key=enc_key,
    )
    assert profile.kaggle_username == "alice"
    assert profile.has_key is True

    # Both files exist
    assert (tmp_path / "profiles" / "default" / "kaggle.key.enc").exists()
    assert (tmp_path / "profiles" / "default" / "profile.json").exists()


def test_key_is_never_plaintext_on_disk(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret-kaggle-key-value",
        root=tmp_path,
        encryption_key=enc_key,
    )
    # No file under the profile dir contains the raw key string.
    for path in (tmp_path / "profiles" / "default").rglob("*"):
        if path.is_file():
            blob = path.read_bytes()
            assert b"secret-kaggle-key-value" not in blob, f"plaintext leaked into {path.name}"


def test_get_kaggle_key_round_trip(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret-kaggle-key-value",
        root=tmp_path,
        encryption_key=enc_key,
    )
    got = profile_store.get_kaggle_key("default", root=tmp_path, encryption_key=enc_key)
    assert got == ("alice", "secret-kaggle-key-value")


def test_get_kaggle_key_returns_none_when_not_set(tmp_path, enc_key):
    got = profile_store.get_kaggle_key("default", root=tmp_path, encryption_key=enc_key)
    assert got is None


def test_wrong_encryption_key_raises(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret",
        root=tmp_path,
        encryption_key=enc_key,
    )
    wrong = Fernet.generate_key().decode()
    with pytest.raises(RuntimeError, match="Cannot decrypt"):
        profile_store.get_kaggle_key("default", root=tmp_path, encryption_key=wrong)


def test_missing_encryption_key_raises_clear_message(tmp_path):
    with pytest.raises(RuntimeError, match="PROFILE_ENCRYPTION_KEY is not set"):
        profile_store.set_kaggle_key(
            "default",
            username="alice",
            key="secret",
            root=tmp_path,
            encryption_key="",
        )


def test_get_profile_never_exposes_key_field(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret",
        root=tmp_path,
        encryption_key=enc_key,
    )
    profile = profile_store.get_profile("default", root=tmp_path)
    dumped = profile.model_dump()
    assert "key" not in dumped and "kaggle_key" not in dumped


def test_clear_removes_both_files(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret",
        root=tmp_path,
        encryption_key=enc_key,
    )
    profile_store.clear_kaggle_key("default", root=tmp_path)
    assert not (tmp_path / "profiles" / "default" / "kaggle.key.enc").exists()
    assert not (tmp_path / "profiles" / "default" / "profile.json").exists()


def test_clear_is_idempotent(tmp_path):
    profile_store.clear_kaggle_key("default", root=tmp_path)
    profile_store.clear_kaggle_key("default", root=tmp_path)  # no error


def test_mark_validated_updates_timestamp(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret",
        root=tmp_path,
        encryption_key=enc_key,
    )
    when = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
    profile_store.mark_validated("default", root=tmp_path, when=when)
    profile = profile_store.get_profile("default", root=tmp_path)
    assert profile.validated_at == when


def test_mark_used_updates_timestamp(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="secret",
        root=tmp_path,
        encryption_key=enc_key,
    )
    when = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
    profile_store.mark_used("default", root=tmp_path, when=when)
    profile = profile_store.get_profile("default", root=tmp_path)
    assert profile.last_used_at == when
