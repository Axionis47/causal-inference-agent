"""Datasets on disk: the index, the files the writer writes, the upload staging area, and the per-dataset meta the
page needs. Facts only."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
import yaml

from causal_agent.profile import data as D
from causal_agent.profile.profiler import Profile, profile
from causal_agent.server.context import render_context
from causal_agent.server.models import ColumnSummary, DatasetCreate, DatasetSummary, DatetimeShape, NumericShape, ProfileOut, Sentinel, TopValue
from causal_agent.server.settings import Settings

HEADER = "# The eval datasets. Name → files and the profiler flags used to build the profile.\n"


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class BadUpload(Exception):
    pass


# ------------------------------------------------------------------ index and meta


def entries(s: Settings) -> dict:
    return (yaml.safe_load(s.index.read_text()) if s.index.exists() else None) or {}


def write_entries(s: Settings, es: dict) -> None:
    s.index.parent.mkdir(parents=True, exist_ok=True)
    s.index.write_text(HEADER + yaml.safe_dump(es, sort_keys=False, allow_unicode=True))


def meta_path(s: Settings, name: str) -> Path:
    return s.web_root / name / "meta.json"


def read_meta(s: Settings, name: str) -> dict | None:
    p = meta_path(s, name)
    return json.loads(p.read_text()) if p.exists() else None


def write_meta(s: Settings, name: str, meta: dict) -> None:
    p = meta_path(s, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, default=str))


def all_meta(s: Settings) -> dict[str, dict]:
    out = {}
    if s.web_root.exists():
        for d in sorted(s.web_root.iterdir()):
            m = read_meta(s, d.name)
            if m:
                out[d.name] = m
    return out


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ list


def _profile_counts(s: Settings, entry: dict) -> tuple[int | None, int | None]:
    p = s.root / entry.get("profile", "")
    if not entry.get("profile") or not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text()).get("dataset") or {}
        return d.get("rows"), d.get("columns")
    except Exception:
        return None, None


def list_datasets(s: Settings) -> list[DatasetSummary]:
    es, metas = entries(s), all_meta(s)
    out = []
    for name in sorted(set(es) | set(metas), key=lambda n: (n not in metas, (metas.get(n) or {}).get("created_at") or "", n), reverse=False):
        e, m = es.get(name) or {}, metas.get(name)
        rows, cols = _profile_counts(s, e)
        out.append(DatasetSummary(name=name, title=(m or {}).get("title") or name.replace("_", " "), csv=e.get("csv") or (m or {}).get("csv") or "",
                                  rows=rows, columns=cols, created_at=(m or {}).get("created_at"), shipped=m is None, has_claims="claims" in e,
                                  question=(m or {}).get("question")))
    # newest web datasets first, then the shipped ones by name
    out.sort(key=lambda d: (d.shipped, -(dt.datetime.fromisoformat(d.created_at).timestamp() if d.created_at else 0), d.name))
    return out


# ------------------------------------------------------------------ upload and profile


def _examples(c) -> list[str]:
    if c.kind in {"categorical", "boolean"} and c.top_values:
        return [t.value for t in c.top_values[:4]]
    if c.kind in {"numeric", "id"} and c.numeric:
        return [f"{c.numeric.min:g}", f"{c.numeric.max:g}"]
    if c.kind == "datetime" and c.datetime:
        return [c.datetime.first, c.datetime.last]
    return []


def _column(c) -> ColumnSummary:
    num = NumericShape(min=c.numeric.min, p25=c.numeric.p25, p50=c.numeric.p50, p75=c.numeric.p75, max=c.numeric.max, mean=c.numeric.mean) if c.numeric else None
    dt = DatetimeShape(first=c.datetime.first, last=c.datetime.last, frequency=c.datetime.inferred_frequency) if c.datetime else None
    return ColumnSummary(
        name=c.name, key=c.key, kind=c.kind, nulls=c.nulls, null_rate=c.null_rate, distinct=c.distinct, constant=c.constant, examples=_examples(c),
        numeric=num, top_values=[TopValue(value=t.value, count=t.count, share=t.share) for t in (c.top_values or [])], datetime=dt,
        sentinels=[Sentinel(value=x.value, count=x.count, reason=x.reason) for x in c.observed_sentinels], issues=list(c.format_issues),
    )


def summarise(prof: Profile) -> list[ColumnSummary]:
    return [_column(c) for c in prof.columns]


HEAD_ROWS = 8


def head_rows(path: Path, n: int = HEAD_ROWS) -> list[list[str]]:
    """The first n rows as strings, in file order. Missing cells are empty strings."""
    df = pd.read_csv(path, nrows=n, dtype=str, keep_default_na=False)
    return [[str(v) for v in row] for row in df.itertuples(index=False, name=None)]


def profile_out(upload_id: str, path: Path, prof: Profile) -> ProfileOut:
    d = prof.dataset
    return ProfileOut(
        upload_id=upload_id, filename=path.name, rows=d.rows, columns=summarise(prof), head=head_rows(path),
        duplicate_rows=d.duplicate_rows, candidate_keys=d.candidate_keys, grain=d.grain, co_missing=d.co_missing, issues=list(d.format_issues),
    )


def _safe_filename(filename: str) -> str:
    base = Path(filename or "upload.csv").name
    keep = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in base).strip() or "upload.csv"
    return keep if keep.lower().endswith(".csv") else keep + ".csv"


def stage_upload(s: Settings, filename: str, content: bytes) -> ProfileOut:
    if not (filename or "").lower().endswith(".csv"):
        raise BadUpload("Only .csv files can be uploaded.")
    if not content.strip():
        raise BadUpload("The file is empty.")
    upload_id = uuid.uuid4().hex[:8]
    d = s.uploads / upload_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / _safe_filename(filename)
    path.write_bytes(content)
    try:
        prof = profile(path)
    except Exception as e:  # pandas could not read it
        shutil.rmtree(d, ignore_errors=True)
        raise BadUpload(f"The file could not be read as a table: {e}") from e
    return profile_out(upload_id, path, prof)


def _staged(s: Settings, upload_id: str) -> Path:
    d = s.uploads / upload_id
    files = sorted(d.glob("*.csv")) if d.exists() else []
    if not files:
        raise NotFound(f"upload {upload_id!r} not found; upload the file again")
    return files[0]


# ------------------------------------------------------------------ create


def create_dataset(s: Settings, req: DatasetCreate) -> tuple[DatasetSummary, dict]:
    if req.name in entries(s) or read_meta(s, req.name):
        raise Conflict(f"a dataset named {req.name!r} already exists")
    src = _staged(s, req.upload_id)
    raw_dir = s.root / "data" / "raw" / req.name
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / src.name
    shutil.move(str(src), dest)
    shutil.rmtree(src.parent, ignore_errors=True)
    prof = profile(dest)
    by_name = {c.name: c for c in prof.columns}
    given = {c.name: c.description for c in req.columns}
    columns = [(c.name, given.get(c.name, "")) for c in prof.columns]
    context = render_context(req.title, req.about, req.changed, columns)
    for sub in ("profiles", "context"):
        (s.root / "data" / sub).mkdir(parents=True, exist_ok=True)
    profile_rel, note_rel, csv_rel = f"data/profiles/{req.name}.json", f"data/context/{req.name}.md", str(dest.relative_to(s.root))
    (s.root / profile_rel).write_text(json.dumps(prof.model_dump(), indent=2))
    (s.root / note_rel).write_text(context)
    es = entries(s)
    es[req.name] = {"csv": csv_rel, "note": note_rel, "profile": profile_rel}
    write_entries(s, es)
    meta = {"name": req.name, "title": req.title.strip(), "created_at": _now(), "csv": csv_rel, "thread_id": None, "question": req.question.strip(),
            "form": {"about": req.about, "changed": req.changed, "columns": [{"name": n, "description": d} for n, d in columns]},
            "context": context, "last_prompt": None, "ended": False, "unknown_columns": sorted(set(given) - set(by_name))}
    write_meta(s, req.name, meta)
    D.clear()
    return DatasetSummary(name=req.name, title=meta["title"], csv=csv_rel, rows=prof.dataset.rows, columns=prof.dataset.columns, created_at=meta["created_at"],
                          shipped=False, has_claims=False, question=meta["question"]), meta


# ------------------------------------------------------------------ delete


def _invalidate(name: str) -> None:
    """The file and profile cache is process-global; the lanes read the hand-off and the routing loads the memory per node, so
    there is no other cache to clear."""
    D.clear()


def delete_dataset(s: Settings, name: str, run_dirs: list[str] | None = None) -> None:
    es, meta = entries(s), read_meta(s, name)
    e = es.get(name)
    if e is None and meta is None:
        raise NotFound(f"no dataset named {name!r}")
    removed = []
    for rel in ((e or {}).get("claims"), (e or {}).get("note"), (e or {}).get("profile")):
        if rel and (s.root / rel).exists():
            (s.root / rel).unlink()
            removed.append(rel)
    for stale in (s.root / "data" / "claims" / f"{name}.yaml", s.root / "data" / "context" / f"{name}.md", s.root / "data" / "profiles" / f"{name}.json"):
        if stale.exists():
            stale.unlink()
    shutil.rmtree(s.root / "data" / "memory" / name, ignore_errors=True)
    csv_rel = (e or {}).get("csv") or (meta or {}).get("csv")
    others = [n for n, x in es.items() if n != name and x.get("csv") == csv_rel]
    if csv_rel and not others and csv_rel.startswith(f"data/raw/{name}/"):
        shutil.rmtree(s.root / "data" / "raw" / name, ignore_errors=True)
    if name in es:
        del es[name]
        write_entries(s, es)
    if (s.web_root / name).exists():
        shutil.rmtree(s.web_root / name, ignore_errors=True)
    root = s.run_root.resolve()
    for rd in run_dirs or []:
        p = Path(rd).resolve()
        if p.exists() and p.is_dir() and p != root and root in p.parents:
            shutil.rmtree(p, ignore_errors=True)
    _invalidate(name)
