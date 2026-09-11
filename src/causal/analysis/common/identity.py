"""Pin the implementation behind a capability, including retained adapter metadata."""
from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from pathlib import Path

from causal.analysis.common.catalog import method_module


def implementation_hash(method: str) -> str:
    module = method_module(method)
    method_root = Path(str(module.__file__)).parent
    root = Path(__file__).resolve().parents[1]
    # Test fixtures never affect a run's identity. Executable source, method
    # guidance and the retained numerical profile do; no estimator is imported.
    paths = [*root.joinpath("methods").glob("*/*.py"), *method_root.glob("guidance/*.md"),
             *root.glob("*.py"), *root.joinpath("common").glob("*.py")]
    paths += [root / "integration" / name for name in (
        "numerical.py", "contracts.py", "packs.py", "resources/method-pack-estimation.v1.json")]
    paths += [root.parent / "shared" / name for name in ("contracts.py", "canonical.py")]
    digest = hashlib.sha256(platform.python_version().encode())
    for path in sorted(paths):
        digest.update(str(path.relative_to(root.parent)).encode())
        digest.update(path.read_bytes())
    # Package versions are read as metadata, without importing their code.
    for library in ("pydantic", "polars", "pandas", "numpy", "scipy", "pyfixest", "scikit-learn", "rdrobust", "rddensity"):
        try:
            version = metadata.version(library)
        except metadata.PackageNotFoundError:
            version = "not installed"
        digest.update(f"{library}:{version}".encode())
    return digest.hexdigest()
