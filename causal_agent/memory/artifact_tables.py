"""Read registered CSV or JSON tables consistently for selection and evidence."""

import csv
import hashlib
import io
import json


def rows(name: str, raw: bytes) -> list[list]:
    text = raw.decode("utf-8")
    if not name.endswith(".json"):
        return list(csv.reader(io.StringIO(text)))
    value = json.loads(text)
    if isinstance(value, list) and all(isinstance(r, dict) for r in value):
        columns = list(dict.fromkeys(k for r in value for k in r))
        return [columns] + [[r.get(k) for k in columns] for r in value]
    if isinstance(value, dict) and isinstance(value.get("columns"), list) and isinstance(value.get("data"), list):
        return [value["columns"], *value["data"]]
    if isinstance(value, dict) and all(isinstance(v, list) for v in value.values()):
        columns = list(value)
        if len({len(v) for v in value.values()}) > 1:
            raise ValueError("JSON table columns have different lengths")
        return [columns, *[list(row) for row in zip(*value.values())]]
    raise ValueError("JSON tables must contain records, column arrays, or columns and data")


def compact_evidence(value):
    """Preserve scientific fields while bounding large row/weight collections."""
    if isinstance(value, (list, dict)) and len(value) > 30:
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
        preview = value[:30] if isinstance(value, list) else dict(list(value.items())[:30])
        return {"preview_only": True, "total": len(value), "sha256": digest, "first_items": compact_evidence(preview), "full_evidence": "result.json"}
    if isinstance(value, dict):
        return {key: compact_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_evidence(item) for item in value]
    return value
