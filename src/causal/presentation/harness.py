# Presentation dependencies and the plumbing the §6 coordinator shares: flush-gated commits,
# committed reads, deterministic §19.1 events, the typed §17.2 destinations, and the §18
# PresentationRun record that is terminal on every exit. Every upstream payload is read as data:
# this module imports no design or estimation code, and PRD-005 keeps no LangGraph state at all.

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, cast

from psycopg import Connection

from causal.presentation import compile as co
from causal.presentation import contracts as pc
from causal.presentation import curate as cu
from causal.presentation import render as rd
from causal.shared import events, persistence
from causal.shared.agenttask import GatewayProtocol
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef
from causal.shared.registry import ArtifactTypeRegistry
from causal.shared.validation import ValidationReport

COMPONENT, VERSION = "presentation-coordinator", "presentation-harness.v1"
MANIFEST, PLAN, SPEC = "PresentationContextManifest", "FigurePlan", "FigureSpec"
RENDER, REPORT, BUNDLE = "PresentationRender", "PresentationValidationReport", "PresentationBundle"
# §17.2 evaluation surfaces, one per §19.1 event boundary this stage owns.
E_ENTRY, E_CURATOR, E_PLAN = ("EV-P5-001",), ("EV-P5-002",), ("EV-P5-003",)
E_COMPILE, E_RENDER, E_BUNDLE = ("EV-P5-004",), ("EV-P5-005",), ("EV-P5-006",)
COMPLETE: Final = "complete"
QUALIFIED: Final = "complete_with_qualifications"
BLOCKED, FAILED, TRACE_FAILED = "blocked", "failed", "failed_observability"
UNCITED, UNFROZEN = "uncited_summary_statement", "unfrozen_bundle_parent"
EXECUTABLE, OWNER = "executable_construct", "presentation-catalog-maintainer"
STAGE, ERROR = "stage-coordinator", events.Severity.ERROR
# §17.2: who owns each typed non-delivering status. This stage routes; it never repairs.
DESTINATIONS: Final[dict[str, str]] = {"needs_template": OWNER, "needs_layout_revision": OWNER,
                                       BLOCKED: STAGE, FAILED: STAGE, TRACE_FAILED: STAGE}
# `stage_runs.run_state` is a closed vocabulary and the run record carries the §1 status, so a
# typed catalog-owner status is still a completed run; only a refusal or a crash is not.
ROW_STATE: Final[dict[str, str]] = {BLOCKED: FAILED, FAILED: FAILED, TRACE_FAILED: TRACE_FAILED}
# The §7.1/§10 registry vector this stage pins on every payload it freezes.
REGISTRY_VERSIONS: Final[dict[str, str]] = {
    "artifact_types": "artifact-types.v1", "schema": "presentation-schemas.v1",
    "prompt": cu.PINNED_VERSIONS["prompt"], "model_profile": cu.PINNED_VERSIONS["model_profile"],
    "validators": "presentation-validators.v1", "compiler": co.COMPILER_VERSION,
    "renderer": rd.RENDERER_VERSION}
# §11.3/§13: any Vega-Lite key that would compute, fetch, or execute rather than draw one frozen
# field. Gate 3 reads the compiled document itself, not the compiler's word for it.
EXECUTABLE_KEYS: Final = ("transform", "url", "signal", "expr", "datasets")
RUN_ROW: Final = ("INSERT INTO presentation.runs VALUES (%s, %s, 'running', %s, NULL, NULL,"
                  " %s::jsonb, %s, %s) ON CONFLICT (stage_run_id) DO UPDATE SET"
                  " state = 'running', updated_at = EXCLUDED.updated_at")
RUN_STATE: Final = ("UPDATE presentation.runs SET state = %s, bundle_artifact_id = %s,"
                    " error_code = %s, run_record = %s::jsonb, updated_at = %s"
                    " WHERE stage_run_id = %s")
_CITE = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class PresentationDeps:
    # Everything the stage needs; nothing here is discovered at run time. `observability_ready` is
    # §5 condition 13 as the caller measured it: this stage's gate never calls a service itself,
    # and `gateway` is the ONE model receiver PRD-005 has — the §9 curator call.

    conn: Connection[Any]
    products: persistence.ProductStore
    objects: persistence.ObjectStore
    committer: persistence.ArtifactCommitter
    registry: ArtifactTypeRegistry
    emitter: events.EventEmitter
    clock: Callable[[], datetime]
    catalog: pc.VisualizationCatalogV1
    repo_root: Path
    render_root: Path
    gateway: GatewayProtocol | None = None
    observability_ready: bool = True


@dataclass(frozen=True)
class PresentationRunResult:
    # What one presentation revision returns (PRD-005 §1, §17.1).

    status: str
    analysis_id: str
    stage_run_id: str
    presentation_revision: int
    presentation_bundle_id: str | None = None
    destination: str | None = None
    error_code: str | None = None
    detail_codes: tuple[str, ...] = ()


def ref(envelope: ArtifactEnvelopeV1) -> ArtifactRef:
    return ArtifactRef(artifact_id=envelope.artifact_id, content_hash=envelope.content_hash)


def document_codes(document: Mapping[str, Any]) -> tuple[str, ...]:
    # Gate 3: the compiled document draws frozen fields and nothing else — no transform, no remote
    # asset, no signal, expression, or second inline dataset anywhere inside it.
    return tuple(f"{EXECUTABLE}:{name}" for name in EXECUTABLE_KEYS
                 if f'"{name}":' in json.dumps(document, sort_keys=True))


def summary_codes(summary: str, allowed: frozenset[str]) -> tuple[str, ...]:
    # §15/gate 5: every substantive sentence cites an approved claim statement, an approved
    # qualification, or a frozen artifact id. Nothing else in the summary carries an assertion.
    seen = {row: {name.strip() for found in _CITE.findall(row) for name in found.split(",")}
            for row in summary.splitlines() if row.strip()}
    return tuple(f"{UNCITED}:{row[:60]}" for row, cited in seen.items()
                 if not cited or cited - allowed)


def unbound(*args: Any, **over: Any) -> ValidationReport:
    # §9.3: the curator binds its own gate-2 plan validator; nothing else may answer it.
    raise pc.PresentationError("gate 2 has no plan validator", cu.SHAPE_INVALID)


def short(text: object) -> str:
    return str(text or "").replace("\n", " ").strip()[:180]


class HarnessBase:
    # Events, flush-gated commits, committed reads, typed destinations, and the run-row lifecycle
    # the presentation coordinator is built on.

    def __init__(self, deps: PresentationDeps, *, analysis_id: str, stage_run_id: str,
                 revision: int) -> None:
        self.deps, self.analysis_id, self.stage_run_id = deps, analysis_id, stage_run_id
        self.revision, self._count, self.summary = revision, 0, ""
        self.state: dict[str, Any] = {"analysis_id": analysis_id, "stage_run_id": stage_run_id,
                                      "design_revision": revision, "corrections": {},
                                      "open_requirement_ids": []}
        self.status: pc.PresentationOutcomeStatus = COMPLETE
        self.error_code: str | None = None
        self.detail_codes: tuple[str, ...] = ()
        self.upstream, self.figure_refs = dict[str, ArtifactRef](), dict[str, ArtifactRef]()
        self.held, self._cache = dict[str, ArtifactRef](), dict[str, dict[str, Any]]()
        self.figures, self.failures = dict[str, dict[str, Any]](), list[str]()
        self.counters = {"tasks": 0, "events": 0, "corrections": 0}
        # Every artifact this run committed, in commit order (§18's current-artifact map), and
        # the catalog, theme, font, display profile, compiler, and renderer this run pins.
        self.trail: list[tuple[str, ArtifactRef]] = []
        self.identities: dict[str, str] = {
            "catalog": deps.catalog.catalog_version, "theme": deps.catalog.theme_version,
            "theme_hash": deps.catalog.theme_hash, "font": deps.catalog.font_id,
            "compiler": co.COMPILER_VERSION, "renderer": rd.RENDERER_VERSION,
            "display_profile": deps.catalog.display_profile.display_profile_version}

    # -- events, flush-gated commits, and committed reads -------------------

    def event(self, name: str, evals: tuple[str, ...], **over: Any) -> events.OperationalEventV1:
        # §19.1: one event with a deterministic id and this stage's own identities. PRD-005 opens
        # no graph thread, so no event this stage emits ever carries one.
        self._count = self.counters["events"] = self._count + 1
        return events.build_event(
            occurred_at_utc=self.deps.clock(), event_name=name, analysis_id=self.analysis_id,
            event_id=f"evt:{self.stage_run_id}:{self._count}", stage=events.Stage.PRESENTATION,
            stage_run_id=self.stage_run_id, component_id=COMPONENT, required_eval_ids=evals,
            component_version=VERSION,
            versions={"registry": "artifact-types.v1", "prompt": VERSION}, **over)

    def emit(self, name: str, evals: tuple[str, ...], **over: Any) -> None:
        self.deps.emitter.emit(self.event(name, evals, **over))

    def commit(self, kind: str, payload: Mapping[str, object],
               parents: tuple[ArtifactEnvelopeV1, ...]) -> ArtifactEnvelopeV1:
        # §18: one immutable artifact through the flush-gated committer; the commit completes only
        # once the trace span it names is acknowledged.
        body, deps = dict(payload), self.deps
        built = persistence.build_envelope(
            deps.registry, kind, body, analysis_id=self.analysis_id, parents=parents,
            stage_run_id=self.stage_run_id, producer_version=VERSION,
            created_at_utc=deps.clock(), producer_component=COMPONENT)
        done = deps.committer.commit(built, body, self.event(
            "artifact.committed", E_BUNDLE, status="committed", artifact_refs=(ref(built),)))
        self.held[kind] = ref(done)
        self.trail.append((kind, self.held[kind]))
        return done

    def payload(self, artifact_id: str) -> dict[str, Any]:
        # One committed payload, loaded for the node that needs it and held for replay.
        if artifact_id not in self._cache:
            self._cache[artifact_id] = json.loads(self.deps.objects.get(
                self.deps.products.load_envelope(artifact_id).payload_locator))
        return self._cache[artifact_id]

    def parent(self, *refs: ArtifactRef) -> tuple[ArtifactEnvelopeV1, ...]:
        return tuple(self.deps.products.load_envelope(row.artifact_id) for row in refs)

    def kept(self, kind: str) -> tuple[ArtifactRef, ...]:
        return tuple(row for name, row in self.trail if name == kind)

    def move(self, *states: str) -> None:
        # A replay of a finished revision re-enters its own terminal state, which SC §4's closed
        # vocabulary forbids; an already-final row is simply left exactly as it stands.
        for state in states:
            with suppress(persistence.PersistenceError):
                self.deps.products.transition_stage_run(self.stage_run_id, state)

    def stop(self, status: str, code: str, details: Sequence[str] = ()) -> None:
        # Enter one typed §17.2 destination with its stable code and a raised blocker (§19.1).
        self.status = cast(pc.PresentationOutcomeStatus, status)
        self.error_code, self.detail_codes = code, tuple(sorted({*details, code}))
        self.failures.append(code)
        self.emit("blocker.raised", E_BUNDLE, severity=ERROR, error_code=code,
                  safe_dimensions={"destination": DESTINATIONS.get(status, "")})

    # -- the §18 run record, terminal on every exit -------------------------

    def open_run(self) -> None:
        # SC §4: the stage run and its presentation row exist before any artifact is committed.
        now = self.deps.clock()
        try:
            self.deps.products.get_stage_run_state(self.stage_run_id)
        except persistence.PersistenceError:
            self.deps.products.create_stage_run(self.stage_run_id, self.analysis_id,
                                                "presentation")
            self.move("tracing_preflight", "running")
        self.deps.conn.execute(RUN_ROW, (
            self.stage_run_id, self.analysis_id, self.revision,
            json.dumps(self.record(None).canonical_payload()), now, now))

    def record(self, bundle: ArtifactRef | None, *, final: bool = False) -> pc.PresentationRunV1:
        # §18: the small run record. The immutable artifacts, not this row, are the truth.
        return pc.PresentationRunV1(
            analysis_id=self.analysis_id, stage_run_id=self.stage_run_id, status=self.status,
            upstream=dict(self.upstream), created_at=self.deps.clock(),
            current={f"{kind}:{index}": row for index, (kind, row) in enumerate(self.trail)},
            identities=self.identities | ({} if bundle is None
                                          else {"handoff_out": bundle.artifact_id}),
            counters=self.counters | {"trace_acknowledgements": len(self.trail)},
            stable_ids={"failures": tuple(self.failures), "trace_acknowledgements": tuple(
                row.artifact_id for _, row in self.trail)},
            completed_at=self.deps.clock() if final else None)

    def close(self, bundle: ArtifactRef | None) -> PresentationRunResult:
        # SC §4: the run row reaches a terminal state on every exit path, and the §1 terminal
        # outcome record is validated even when nothing was delivered. PRD-005 registers no
        # outcome artifact type, so that validated record travels inside the run row itself.
        outcome = pc.PresentationOutcomeV1(
            status=self.status, stage_run_id=self.stage_run_id, error_code=self.error_code,
            context_manifest=self.held.get(MANIFEST), presentation_bundle=bundle,
            detail_codes=self.detail_codes)
        found = None if bundle is None else bundle.artifact_id
        state = ROW_STATE.get(self.status, "completed")
        self.move(*(("committing", state) if state == "completed" else (state,)))
        # The row keeps PRD-005 §1's own status; `stage_runs` keeps the closed run vocabulary.
        self.deps.conn.execute(RUN_STATE, (
            self.status, found, self.error_code, json.dumps(
                self.record(bundle, final=True).canonical_payload()
                | {"outcome": outcome.canonical_payload()}), self.deps.clock(), self.stage_run_id))
        self.emit("stage.completed" if found else "stage.failed", E_BUNDLE, status=self.status,
                  severity=events.Severity.INFO if found else ERROR, error_code=self.error_code)
        return PresentationRunResult(
            status=self.status, analysis_id=self.analysis_id, stage_run_id=self.stage_run_id,
            presentation_revision=self.revision, destination=DESTINATIONS.get(self.status),
            presentation_bundle_id=found, error_code=self.error_code,
            detail_codes=self.detail_codes)
