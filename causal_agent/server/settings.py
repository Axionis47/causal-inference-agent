"""Where the server reads and writes. Everything hangs off one root so tests can point it at a temp directory."""

from __future__ import annotations

from pathlib import Path

from causal_agent.common import config


class Settings:
    def __init__(
        self,
        root: Path | str | None = None,
        run_root: Path | str | None = None,
        web_root: Path | str | None = None,
        uploads: Path | str | None = None,
        checkpoint_db: Path | str | None = None,
        dist: Path | str | None = None,
    ) -> None:
        self.root: Path = Path(root or config.ROOT)
        paths = config.load(root=self.root).paths
        self.run_root: Path = Path(run_root) if run_root else paths.runs  # .artifacts/runs (or RUN_DIR)
        self.web_root: Path = Path(web_root) if web_root else paths.web  # data/web/<name>/meta.json, transcript.jsonl
        self.uploads: Path = Path(uploads) if uploads else paths.uploads  # .artifacts/web/uploads/<id>/
        self.checkpoint_db: Path = Path(checkpoint_db) if checkpoint_db else paths.checkpoint_db  # .artifacts/web/checkpoints.sqlite
        self.dist: Path = Path(dist) if dist else paths.dist  # web/dist

    @property
    def index(self) -> Path:
        return self.root / "data" / "datasets.yaml"
