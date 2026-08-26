"""T-020: `evals/catalog.v1.yaml` matches the SC SS10.5.1 and PRD evaluation tables.

The catalog is authored as JSON-syntax content (D-010: the pinned stack has no YAML
parser), so `json.loads` is the only reader required here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATALOG = json.loads((REPO / "evals" / "catalog.v1.yaml").read_text(encoding="utf-8"))
ROWS: list[dict[str, object]] = CATALOG["registrations"]
IDS = [str(row["eval_id"]) for row in ROWS]

ID_PATTERN = re.compile(r"^EV-(SYS|E2E|P1|P2|P3)-\d+$")
SOURCE_ID_PATTERN = re.compile(r"EV-[A-Z0-9]+-\d+")
KINDS = frozenset({"contract", "policy", "numerical", "render", "end_to_end"})
FIELDS = frozenset({
    "eval_id", "kind", "owner", "boundary", "fixture_focus", "trigger",
    "hard_pass_condition", "case_budget", "case_notes",
})
EXPECTED: dict[str, tuple[str, ...]] = {
    "EV-SYS": (
        "EV-SYS-001", "EV-SYS-002", "EV-SYS-003", "EV-SYS-004", "EV-SYS-005",
        "EV-SYS-006", "EV-SYS-007",
    ),
    "EV-E2E": (
        "EV-E2E-001", "EV-E2E-002", "EV-E2E-003", "EV-E2E-004", "EV-E2E-005",
        "EV-E2E-006", "EV-E2E-007",
    ),
    "EV-P1": ("EV-P1-001", "EV-P1-002", "EV-P1-003", "EV-P1-004", "EV-P1-005"),
    "EV-P2": (
        "EV-P2-001", "EV-P2-002", "EV-P2-003", "EV-P2-004", "EV-P2-005",
        "EV-P2-006", "EV-P2-007", "EV-P2-008",
    ),
    "EV-P3": (
        "EV-P3-001", "EV-P3-002", "EV-P3-003", "EV-P3-004", "EV-P3-005",
        "EV-P3-006", "EV-P3-007",
    ),
}


def _surface(eval_id: str) -> str:
    return eval_id.rsplit("-", 1)[0]


def test_every_id_is_unique_and_well_formed() -> None:
    assert len(set(IDS)) == len(IDS)
    assert [eval_id for eval_id in IDS if not ID_PATTERN.match(eval_id)] == []


def test_surface_id_sets_match_the_contract_and_prd_tables() -> None:
    registered: dict[str, tuple[str, ...]] = {
        surface: tuple(e for e in IDS if _surface(e) == surface) for surface in EXPECTED
    }
    assert registered == EXPECTED
    assert len(IDS) == sum(len(ids) for ids in EXPECTED.values())


def test_every_row_carries_the_registration_field_set() -> None:
    assert [row for row in ROWS if set(row) != FIELDS] == []


def test_kind_is_in_the_closed_contract_set() -> None:
    assert {str(row["kind"]) for row in ROWS} <= KINDS
    assert all(
        (row["kind"] == "end_to_end") == (_surface(str(row["eval_id"])) == "EV-E2E")
        for row in ROWS
    )


def test_case_budget_respects_the_bounded_policy() -> None:
    for row in ROWS:
        budget = int(str(row["case_budget"]))
        if _surface(str(row["eval_id"])) == "EV-E2E":
            assert budget == 1, row["eval_id"]
        else:
            assert 1 <= budget <= 8, row["eval_id"]


def test_required_eval_ids_emitted_by_source_are_registered() -> None:
    referenced: set[str] = set()
    for path in sorted((REPO / "src" / "causal").rglob("*.py")):
        referenced.update(SOURCE_ID_PATTERN.findall(path.read_text(encoding="utf-8")))
    assert referenced, "no required_eval_ids found under src/causal"
    assert sorted(referenced - set(IDS)) == []
