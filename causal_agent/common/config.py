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
    uploads: Path  # staged uploads before a dataset is made
    checkpoint_db: Path  # the desk's sqlite checkpoints
    dist: Path  # the built web page the API serves

    @property
    def index(self) -> Path:
        return self.data / "datasets.yaml"


@dataclass(frozen=True)
class Model:
    name: str
    project: str | None  # None: the active gcloud project
    location: str
    include_thoughts: bool
    thinking_budget: int


@dataclass(frozen=True)
class Config:
    paths: Paths
    model: Model
    width_budget: int  # columns beyond this trigger the prefilter fan-out
    port: int


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
        uploads=artifacts / "web" / "uploads",
        checkpoint_db=artifacts / "web" / "checkpoints.sqlite",
        dist=root / "web" / "dist",
    )
    model = Model(
        name=env.get("GEMINI_MODEL", "gemini-2.5-flash"),
        project=env.get("VERTEX_PROJECT") or None,
        location=env.get("VERTEX_LOCATION", "us-central1"),
        include_thoughts=_flag(env, "INCLUDE_THOUGHTS", True),
        thinking_budget=int(env.get("THINKING_BUDGET", "1024")),
    )
    return Config(paths=paths, model=model, width_budget=int(env.get("FRAME_WIDTH_BUDGET", "150")), port=int(env.get("PORT", "8000")))


def get() -> Config:
    return load()
