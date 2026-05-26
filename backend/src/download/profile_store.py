"""Per-profile Kaggle credential storage.

The Kaggle key never appears in plaintext on disk. Username + key are
Fernet-encrypted with a server secret (settings.profile_encryption_key)
and written to {root}/profiles/{profile_id}/kaggle.key.enc. The
non-secret profile (username, validated_at, last_used_at) is written
alongside as profile.json.

set_kaggle_key, get_profile, and clear_kaggle_key are the public surface.
get_kaggle_key is internal to the download module (auth.py uses it to
inject env vars when calling Kaggle).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from src.domain.download import KaggleProfile


def _profile_dir(root: Path, profile_id: str) -> Path:
    return Path(root) / "profiles" / profile_id


def _key_path(root: Path, profile_id: str) -> Path:
    return _profile_dir(root, profile_id) / "kaggle.key.enc"


def _profile_path(root: Path, profile_id: str) -> Path:
    return _profile_dir(root, profile_id) / "profile.json"


def _fernet(encryption_key: str) -> Fernet:
    if not encryption_key:
        raise RuntimeError(
            "PROFILE_ENCRYPTION_KEY is not set. Generate one with: "
            "python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\" and add it to backend/.env."
        )
    return Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def set_kaggle_key(
    profile_id: str,
    username: str,
    key: str,
    *,
    root: Path,
    encryption_key: str,
) -> KaggleProfile:
    """Encrypt and persist the Kaggle credentials for profile_id."""
    if not username or not key:
        raise ValueError("username and key are required")

    blob = json.dumps({"username": username, "key": key}).encode()
    _atomic_write(_key_path(root, profile_id), _fernet(encryption_key).encrypt(blob))

    profile = KaggleProfile(
        profile_id=profile_id,
        kaggle_username=username,
        has_key=True,
        validated_at=None,
        last_used_at=None,
    )
    _atomic_write(
        _profile_path(root, profile_id),
        profile.model_dump_json(indent=2).encode(),
    )
    return profile


def get_kaggle_key(
    profile_id: str,
    *,
    root: Path,
    encryption_key: str,
) -> tuple[str, str] | None:
    """Return (username, key) for profile_id, or None if not set."""
    path = _key_path(root, profile_id)
    if not path.exists():
        return None
    try:
        blob = _fernet(encryption_key).decrypt(path.read_bytes())
    except InvalidToken as e:
        raise RuntimeError(
            f"Cannot decrypt profile {profile_id}; encryption key changed?"
        ) from e
    payload = json.loads(blob)
    return payload["username"], payload["key"]


def get_profile(profile_id: str, *, root: Path) -> KaggleProfile | None:
    """Return the non-secret profile record, or None if not set."""
    path = _profile_path(root, profile_id)
    if not path.exists():
        return None
    return KaggleProfile.model_validate_json(path.read_bytes())


def clear_kaggle_key(profile_id: str, *, root: Path) -> None:
    """Remove the encrypted key blob and the profile record."""
    for path in (_key_path(root, profile_id), _profile_path(root, profile_id)):
        if path.exists():
            path.unlink()


def _update_profile(
    profile_id: str,
    *,
    root: Path,
    field: str,
    value: datetime,
) -> None:
    profile = get_profile(profile_id, root=root)
    if profile is None:
        return
    updated = profile.model_copy(update={field: value})
    _atomic_write(
        _profile_path(root, profile_id),
        updated.model_dump_json(indent=2).encode(),
    )


def mark_validated(profile_id: str, *, root: Path, when: datetime | None = None) -> None:
    _update_profile(
        profile_id,
        root=root,
        field="validated_at",
        value=when or datetime.now(timezone.utc),
    )


def mark_used(profile_id: str, *, root: Path, when: datetime | None = None) -> None:
    _update_profile(
        profile_id,
        root=root,
        field="last_used_at",
        value=when or datetime.now(timezone.utc),
    )
