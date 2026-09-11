"""Copy and verify exact committed deliverables; historical bundles remain readable."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

OCCUPIED = "export_directory_occupied"
EXPORT_MISMATCH = "export_hash_mismatch"


class DeliveryError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _filename(prefix: str, name: str) -> str:
    filename = "report.html" if prefix == "report" and name == "html" else f"{prefix}.{name}"
    if prefix == "report" and name.endswith(("_png", "_svg")):
        filename = "report." + name[:-4] + "." + name[-3:]
    if Path(filename).name != filename or "\\" in filename:
        raise DeliveryError("invalid frozen export filename", EXPORT_MISMATCH)
    return filename


def _artifacts(body: Any) -> list[tuple[str, dict[str, Any]]]:
    artifacts = [(str(row.get("visual_id", row.get("figure_id"))), row) for row in body["figures"]]
    if "delivery" in body:
        artifacts.append(("report", body["delivery"]))
    names = {"summary.txt"}
    for prefix, artifact in artifacts:
        if set(artifact["objects"]) != set(artifact["object_hashes"]):
            raise DeliveryError("export object and hash keys differ", EXPORT_MISMATCH)
        for key in artifact["objects"]:
            filename = _filename(prefix, key)
            if filename in names:
                raise DeliveryError(f"duplicate export filename: {filename}", EXPORT_MISMATCH)
            names.add(filename)
    return artifacts


def _verified_bytes(path: Path, expected: str) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DeliveryError(f"{path} cannot be read", EXPORT_MISMATCH) from error
    if hashlib.sha256(raw).hexdigest() != expected:
        raise DeliveryError(f"{path} does not verify", EXPORT_MISMATCH)
    return raw


def verify_export(body: Any, target: Path) -> dict[str, Any]:
    """Reopen every expected copied file; return one manifest for historical and current exports."""
    _verified_bytes(target / "summary.txt", hashlib.sha256(str(body["summary"]).encode()).hexdigest())
    result: dict[str, Any] = {"figures": []}
    for prefix, artifact in _artifacts(body):
        assets = {}
        for key, digest in artifact["object_hashes"].items():
            copied = target / _filename(prefix, key)
            _verified_bytes(copied, digest)
            assets[key] = str(copied.resolve())
        record = {"assets": assets, "object_hashes": artifact["object_hashes"]}
        if "delivery" in body and artifact is body["delivery"]:
            result["report"] = record | {key: artifact[key] for key in ("preview_keys", "draft_hash")
                                         if key in artifact}
        else:
            result["figures"].append({"figure_id": prefix, **record})
    return result


def export(body: Any, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        raise DeliveryError(f"{target} is not empty", OCCUPIED)
    artifacts = _artifacts(body)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.txt").write_text(str(body["summary"]), encoding="utf-8")
    for prefix, artifact in artifacts:
        for name, source in sorted(artifact["objects"].items()):
            raw = _verified_bytes(Path(source), artifact["object_hashes"][name])
            (target / _filename(prefix, name)).write_bytes(raw)
    verify_export(body, target)
