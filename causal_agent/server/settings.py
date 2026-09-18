"""Where the server reads and writes. Everything hangs off one root so tests can point it at a temp directory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from causal_agent.profile.datasets import ROOT


@dataclass
class Settings:
    root: Path = field(default_factory=lambda: ROOT)
    run_root: Path | None = None  # .artifacts/runs (or RUN_DIR)
    web_root: Path | None = None  # data/web/<name>/meta.json, transcript.jsonl
    uploads: Path | None = None  # .artifacts/web/uploads/<id>/
    checkpoint_db: Path | None = None  # .artifacts/web/checkpoints.sqlite
    dist: Path | None = None  # web/dist

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.run_root = Path(self.run_root) if self.run_root else self.root / os.getenv("RUN_DIR", ".artifacts/runs")
        self.web_root = Path(self.web_root) if self.web_root else self.root / "data" / "web"
        self.uploads = Path(self.uploads) if self.uploads else self.root / ".artifacts" / "web" / "uploads"
        self.checkpoint_db = Path(self.checkpoint_db) if self.checkpoint_db else self.root / ".artifacts" / "web" / "checkpoints.sqlite"
        self.dist = Path(self.dist) if self.dist else self.root / "web" / "dist"

    @property
    def index(self) -> Path:
        return self.root / "data" / "datasets.yaml"
