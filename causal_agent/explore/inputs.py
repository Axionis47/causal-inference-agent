"""Materialize only explicitly supplied immutable resources, without CSV projection."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from causal_agent.common import config, trace
from causal_agent.explore.contracts import ArtifactInput, DatasetInput, ExploreRequest

FORMATS = {".csv", ".tsv", ".json", ".png", ".jpg", ".jpeg", ".txt"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_file(path: Path, root: Path, limit: int) -> Path:
    """No links, special files, escapes or unbounded host reads."""
    if path.is_symlink() or not path.is_relative_to(root):
        raise ValueError("file is outside the supplied resource directory")
    for parent in path.relative_to(root).parents:
        if (root / parent).is_symlink():
            raise ValueError("linked resource directory")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
        raise ValueError("resource must be a bounded regular file without links")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("resource escapes its directory")
    return path


def prepare(req: ExploreRequest) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    root = req.out_dir / "input"
    root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, dict[str, str]] = {}
    ids: set[str] = set()
    resources: list[DatasetInput | ArtifactInput] = [req.dataset_version, *req.artifact_refs]
    limit = config.ingestion().max_bytes
    total = 0
    for n, resource in enumerate(resources):
        path = resource.path.absolute()
        safe_file(path, path.parent, limit)
        suffix = path.suffix.lower()
        if suffix not in FORMATS or n == 0 and suffix != ".csv":
            raise ValueError(f"unsupported supplied format: {suffix}; dataset must be CSV")
        if resource.id in ids:
            raise ValueError(f"duplicate supplied resource: {resource.id}")
        ids.add(resource.id)
        data = path.read_bytes()
        total += len(data)
        if total > limit * 4:
            raise ValueError("supplied resources exceed the task input limit")
        if hashlib.sha256(data).hexdigest() != resource.sha256:
            raise ValueError(f"supplied resource checksum changed: {resource.id}")
        name = "dataset.csv" if n == 0 else f"artifact_{n}{suffix}"
        trace.write(root / name, data)
        manifest[name] = {"id": resource.id, "sha256": resource.sha256, "format": suffix}
    trace.write(root / "manifest.json", json.dumps(manifest, indent=2))
    return root / "dataset.csv", root, manifest
