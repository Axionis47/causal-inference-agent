"""Tests for archive safety admission (T-007; PRD-001 §5.5; EV-P1-003 unit layer)."""

from __future__ import annotations

import io
import zipfile

import pytest

from causal.intake.archive import ArchiveSafety


def build_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class TestClassification:
    def test_supported_classes(self) -> None:
        admission = ArchiveSafety().admit(
            build_zip(
                {
                    "data/nsw.csv": b"a,b\n1,2\n",
                    "readme.md": b"# about",
                    "codebook.json": b"{}",
                    "chart.pdf": b"%PDF",
                    "explore.ipynb": b"{}",
                }
            )
        )
        assert admission.safe
        by_name = {d.name: d.classification for d in admission.decisions}
        assert by_name == {
            "data/nsw.csv": "table",
            "readme.md": "document",
            "codebook.json": "metadata",
            "chart.pdf": "unreadable",
            "explore.ipynb": "withheld",
        }

    def test_extract_only_admitted_classes(self) -> None:
        safety = ArchiveSafety()
        data = build_zip({"t.csv": b"a\n1\n", "n.ipynb": b"{}", "img.png": b"x"})
        admission = safety.admit(data)
        extracted = safety.extract_admitted(data, admission)
        assert set(extracted) == {"t.csv"}


class TestSafety:
    @pytest.mark.parametrize(
        ("name", "reason_fragment"),
        [("../evil.csv", "traversal"), ("/abs.csv", "absolute")],
    )
    def test_traversal_and_absolute_paths_unsafe(self, name: str, reason_fragment: str) -> None:
        admission = ArchiveSafety().admit(build_zip({name: b"x", "ok.csv": b"a\n"}))
        assert not admission.safe
        unsafe = next(d for d in admission.decisions if d.classification == "unsafe")
        assert reason_fragment in (unsafe.reason or "")

    @pytest.mark.parametrize("name", ["model.pkl", "tool.exe", "inner.zip", "macro.xlsm"])
    def test_unsafe_extensions(self, name: str) -> None:
        admission = ArchiveSafety().admit(build_zip({name: b"x"}))
        assert not admission.safe

    def test_unsafe_archive_never_extracts(self) -> None:
        safety = ArchiveSafety()
        data = build_zip({"model.pkl": b"x", "t.csv": b"a\n"})
        admission = safety.admit(data)
        with pytest.raises(ValueError, match="failed admission"):
            safety.extract_admitted(data, admission)

    def test_corrupt_archive_refused(self) -> None:
        admission = ArchiveSafety().admit(b"this is not a zip")
        assert not admission.safe and admission.refusal_reason is not None

    def test_bomb_ratio_refused(self) -> None:
        compressible = b"\x00" * (4 * 1024 * 1024)
        admission = ArchiveSafety(max_ratio=50, ratio_floor_bytes=1).admit(
            build_zip({"zeros.csv": compressible})
        )
        assert not admission.safe
        assert "ratio" in (admission.decisions[0].reason or "")

    def test_total_size_limit(self) -> None:
        admission = ArchiveSafety(max_total_bytes=10).admit(build_zip({"t.csv": b"a" * 100}))
        assert not admission.safe and "total" in (admission.refusal_reason or "")

    def test_entry_count_limit(self) -> None:
        entries = {f"f{i}.csv": b"a\n" for i in range(5)}
        admission = ArchiveSafety(max_entries=3).admit(build_zip(entries))
        assert not admission.safe

    def test_every_entry_gets_a_decision(self) -> None:
        data = build_zip({"a.csv": b"x\n", "b.unknownext": b"y", "c.md": b"z"})
        admission = ArchiveSafety().admit(data)
        assert len(admission.decisions) == 3  # no file silently disappears
