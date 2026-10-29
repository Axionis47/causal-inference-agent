"""One reading of the environment. Every path hangs off the repository root; every knob has a default that works with no
`.env` at all. `get()` reads the environment each time it is called, so a test that sets a variable sees it at once and
nothing needs resetting."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path  # data/: the index, profiles, claims, context, memory
    memory: Path  # data/memory/<name>/
    web: Path  # data/web/<name>/meta.json, transcript.jsonl
    runs: Path  # where a lane writes a run's table, design, estimates, report
    profiles: Path  # the profile cache, keyed by file digest
    checkpoint_db: Path  # the desk's sqlite checkpoints
    dist: Path  # the built web page the API serves

    @property
    def index(self) -> Path:
        return self.data / "datasets.yaml"


@dataclass(frozen=True)
class Model:
    name: str  # the heavy tier: every judgement that designs, decides or explains
    light: str  # the light tier: the per-turn reading of the person's words; the heavy model unless GEMINI_MODEL_LIGHT says otherwise
    project: str | None  # None: the active gcloud project
    location: str
    include_thoughts: bool
    thinking_budget: int


@dataclass(frozen=True)
class Sandbox:
    image: str  # the mandatory Docker script runtime
    queue: str = ""  # Compose-only private spool, serviced by the network-disabled Docker worker
    agent_url: str = ""  # persistent A2A service; empty starts a local trusted agent process


@dataclass(frozen=True)
class Decider:
    kind: str  # laya | jev: the decision model that reads each turn's kind; laya runs in this process, jev is a hosted endpoint on the same wire
    url: str | None  # the endpoint, when jev
    key: str | None  # its key, when jev


@dataclass(frozen=True)
class Config:
    paths: Paths
    model: Model
    width_budget: int  # columns beyond this trigger the prefilter fan-out
    port: int
    sandbox: Sandbox
    decider: Decider


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    v = env.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def load(env: Mapping[str, str] | None = None, root: Path | None = None) -> Config:
    """A Config from an environment mapping. Pure: nothing is read from disk, nothing is cached."""
    env = os.environ if env is None else env
    root = Path(root or ROOT)
    artifacts = root / ".artifacts"
    paths = Paths(
        root=root,
        data=root / "data",
        memory=root / "data" / "memory",
        web=root / "data" / "web",
        runs=root / env.get("RUN_DIR", ".artifacts/runs"),
        profiles=root / env.get("PROFILE_CACHE_DIR", ".artifacts/profiles"),
        checkpoint_db=artifacts / "web" / "checkpoints.sqlite",
        dist=root / "web" / "dist",
    )
    heavy = env.get("GEMINI_MODEL", "gemini-2.5-flash")
    model = Model(
        name=heavy,
        light=env.get("GEMINI_MODEL_LIGHT") or heavy,
        project=env.get("VERTEX_PROJECT") or None,
        location=env.get("VERTEX_LOCATION", "us-central1"),
        include_thoughts=_flag(env, "INCLUDE_THOUGHTS", True),
        thinking_budget=int(env.get("THINKING_BUDGET", "1024")),
    )
    sandbox = Sandbox(image=env.get("EXPLORE_IMAGE", "causal-desk-explore"), queue=env.get("EXPLORE_QUEUE", ""), agent_url=env.get("EXPLORE_AGENT_URL", ""))
    decider = Decider(kind=env.get("DECIDER", "laya"), url=env.get("DECIDER_URL") or None, key=env.get("DECIDER_KEY") or None)
    return Config(
        paths=paths, model=model, width_budget=int(env.get("FRAME_WIDTH_BUDGET", "150")), port=int(env.get("PORT", "8000")), sandbox=sandbox, decider=decider
    )


def get() -> Config:
    return load()


@dataclass(frozen=True)
class Ingestion:
    database_url: str
    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    user_id: str
    mode: str
    max_bytes: int
    max_rows: int
    max_columns: int
    max_cells: int


def ingestion(env: Mapping[str, str] | None = None) -> Ingestion:
    env = os.environ if env is None else env
    return Ingestion(
        database_url=env.get("DATABASE_URL", ""),
        endpoint=env.get("S3_ENDPOINT", "http://localhost:8333"),
        bucket=env.get("S3_BUCKET", "datasets"),
        access_key=env.get("S3_ACCESS_KEY", ""),
        secret_key=env.get("S3_SECRET_KEY", ""),
        user_id=env.get("DEV_USER_ID", "00000000-0000-0000-0000-000000000001"),
        mode=env.get("APP_ENV", "development"),
        max_bytes=int(env.get("UPLOAD_MAX_BYTES", str(25 * 1024 * 1024))),
        max_rows=int(env.get("UPLOAD_MAX_ROWS", "100000")),
        max_columns=int(env.get("UPLOAD_MAX_COLUMNS", "200")),
        max_cells=int(env.get("UPLOAD_MAX_CELLS", "5000000")),
    )
