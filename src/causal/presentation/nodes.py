# The §6 presentation coordinator: the thirteen-condition §5 entry gate, the frozen §7.1 context
# manifest, the ONE §9 curator call, sequential §13 compilation and rendering in plan order, the
# §15 cited summary, and the one committed PresentationBundle. The five §16 gates run in order
# and no later gate ever waives an earlier failure. This is a plain sequential coordinator — no
# LangGraph, no checkpointer, no interrupt — and every upstream payload arrives here as data.

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from causal.presentation import catalog as vc
from causal.presentation import compile as co
from causal.presentation import contracts as pc
from causal.presentation import curate as cu
from causal.presentation import harness as ph
from causal.presentation import render as rd
from causal.shared import handoff, persistence
from causal.shared.agenttask import TaskRunner
from causal.shared.contracts import ArtifactRef, HandoffManifestV1


class PresentationCoordinator(ph.HarnessBase):
    # Entry and the frozen context manifest, the one curator call, compile and render in plan
    # order, the §15 summary, the final validation report, and the one committed bundle.

    # -- gate 1: the thirteen §5 conditions over the PRD-004 handoff --------

    def entry(self, incoming: HandoffManifestV1, outcome: ArtifactRef,
              view: ArtifactRef) -> pc.MethodProfileV1 | None:
        self.emit("stage.started", ph.E_ENTRY)
        self.emit("task.started", ph.E_ENTRY, status="entry")
        self.upstream = dict(zip(pc.ENTRY_KEYS, incoming.entries, strict=True))
        self.identities |= {"handoff_in": incoming.handoff_id, "outcome": outcome.artifact_id}
        accepted, profile = self._accepted(incoming), None
        try:
            design = self.payload(self.upstream["experiment_design"].artifact_id)
            profile = vc.profile_for_method(self.deps.catalog, str(design["method_id"]))
            codes = vc.entry_codes(self._inputs(outcome, view, design), vc.EntryPolicy(
                catalog=self.deps.catalog, profile=profile, approved_graph_view=view,
                observability_ready=self.deps.observability_ready))
        except (persistence.PersistenceError, pc.PresentationError, KeyError, TypeError) as bad:
            profile, codes = None, (str(getattr(bad, "code", vc.ENTRY_VALIDATION_FAILED)),)
        if codes := codes if accepted else (*codes, vc.ENTRY_VALIDATION_FAILED):
            self.emit("task.failed", ph.E_ENTRY, severity=ph.ERROR, error_code=codes[0])
            self.stop(ph.BLOCKED, codes[0], codes)
            return None
        self.emit("task.completed", ph.E_ENTRY, status="entry")
        return profile

    def _accepted(self, incoming: HandoffManifestV1) -> bool:
        # T-006/D-037: load first, then check — a recorded handoff replays its acceptance instead
        # of raising `duplicate_handoff` when the same PRD-004 boundary is presented twice.
        deps, store = self.deps, handoff.HandoffStore(self.deps.conn)
        try:
            found = store.load(incoming.handoff_id)
        except persistence.PersistenceError:
            return handoff.HandoffGate(
                deps.objects, deps.products, store, deps.registry, deps.emitter).accept(
                incoming, ph.COMPONENT, frozenset({ph.COMPLETE}),
                lambda verdict, codes: self.event(
                    f"handoff.{verdict}", ph.E_ENTRY, status=verdict,
                    error_code=codes[0] if codes else None)).accepted
        return found.receiver_validation_result == "accepted" and found.entries == incoming.entries

    def _inputs(self, outcome: ArtifactRef, view: ArtifactRef,
                design: Mapping[str, Any]) -> vc.EntryInputs:
        # §5 condition 6 measures every claim citation against the estimation bundle's own frozen
        # refs, and each evidence result hangs one level below it, so those are resolved too.
        up, held = self.upstream, self.deps.products.find_artifact_hash
        frozen = dict(self.payload(up["estimation_bundle"].artifact_id))
        for row in frozen.get("evidence_bundles") or ():
            for one in self.payload(str(row["artifact_id"])).get("results") or ():
                frozen[f"evidence_result:{one['artifact_id']}"] = one
        figures: dict[str, dict[str, Any]] = {}
        for row in self.payload(up["figure_data_bundle"].artifact_id)["results"]:
            # Two builders may answer one question; the first in bundle order is the one drawn.
            body = self.payload(str(row["artifact_id"]))
            if figures.setdefault(str(body["visual_evidence_id"]), body) is body:
                self.figure_refs[str(body["visual_evidence_id"])] = ArtifactRef(**dict(row))
        self.figures = figures
        return vc.EntryInputs(
            outcome=self.payload(outcome.artifact_id), bundle=frozen, design=design,
            judgment=self.payload(up["claim_judgment"].artifact_id), declared=up,
            capacity=self.payload(up["capacity_check"].artifact_id), figure_data=figures,
            graph_view={"artifact_id": view.artifact_id, "content_hash": held(view.artifact_id)},
            committed={key: None if (digest := held(row.artifact_id)) is None else ArtifactRef(
                artifact_id=row.artifact_id, content_hash=digest) for key, row in up.items()})

    # -- the frozen §7.1 presentation-context manifest ----------------------

    def freeze(self, profile: pc.MethodProfileV1, outcome: ArtifactRef,
               view: ArtifactRef) -> pc.PresentationContextManifestV1:
        # §7.1: the one authoritative presentation-context surface, frozen before the curator
        # runs. It copies the approved selection verbatim and states no fact of its own. The
        # incoming handoff is a visibility row, not an artifact; the estimation outcome it was
        # rebuilt from (D-037) is the committed artifact that stands for it here.
        up, cat = self.upstream, self.deps.catalog
        judgment = self.payload(up["claim_judgment"].artifact_id)
        book = pc.PresentationContextManifestV1(
            parents=(up["estimation_bundle"],), versions=dict(ph.REGISTRY_VERSIONS) | {
                "visualization_catalog": cat.catalog_version, "theme": cat.theme_version,
                "display_profile": cat.display_profile.display_profile_version,
                "font": cat.font_id},
            analysis_id=self.analysis_id, stage_run_id=self.stage_run_id, inputs=dict(up),
            handoff_manifest=outcome, causal_graph_view=view, display_profile=cat.display_profile,
            approved={"question_id": up["experiment_design"].artifact_id, "method_id":
                      profile.method_id, "estimand_id": str(judgment["estimand_id"]),
                      "profile_id": profile.profile_id},
            required_evidence_ids=profile.required_evidence_ids, claim_status=judgment["status"],
            allowlists={"artifact_ids": tuple(sorted(
                {row.artifact_id for row in (*up.values(), *self.figure_refs.values())}
                | {view.artifact_id, outcome.artifact_id}))},
            statement_ids=tuple(str(row["contrast_id"]) for row in judgment["items"]),
            qualification_ids=tuple(str(row) for row in judgment.get("qualifications") or ()),
            evidence=tuple(self._evidence(name, profile) for name in profile.evidence_order
                           if name in self.figure_refs))
        self.commit(ph.MANIFEST, book.canonical_payload(), self.parent(*up.values()))
        return book

    def _evidence(self, name: str, profile: pc.MethodProfileV1) -> pc.EvidenceEntryV1:
        # §7.1: bounded facts about one frozen figure-data artifact — never an observation.
        body = self.payload(self.figure_refs[name].artifact_id)
        units = {role: str(unit) for role, unit in (body.get("units") or {}).items()}
        return pc.EvidenceEntryV1(
            visual_evidence_id=name, figure_data=self.figure_refs[name], units=units,
            figure_data_schema_id=str(body["schema_version"]),
            compatible_template_ids=profile.templates_by_evidence[name],
            quantities={role: f"{name}.{role}" for role in units},
            cardinality=len(body.get("points") or ()),
            suppression_state=str(body["disclosure_status"]))

    # -- gate 2: the ONE §9 curator call ------------------------------------

    def curate(self, book: pc.PresentationContextManifestV1,
               profile: pc.MethodProfileV1) -> pc.FigurePlanV1 | None:
        # §9: one bounded call, at most two targeted corrections, one allowlisted layout-fact
        # operation. D-071 keeps figure identity, coverage, lineage, and versions on this side.
        held = self.held[ph.MANIFEST]
        parents = self.parent(held, self.upstream["claim_judgment"], *self.figure_refs.values())
        runner = TaskRunner(
            gateway=cast(Any, self.deps.gateway), tasks={cu.TASK_KIND: cu.CURATOR_TASK},
            tools={cu.TASK_KIND: (cu.TOOL_ID,)}, evals={cu.TASK_KIND: ph.E_CURATOR},
            prompts_root=self.deps.repo_root, envelope=cu.build_curator_envelope,
            prompt=cu.render_curator_prompt, validate=ph.unbound, context=lambda state: None,
            manifest=lambda state: held, upsert=lambda *args: None,
            parents=lambda state, *kinds: parents,
            evidence=lambda state: frozenset(book.allowlists["artifact_ids"]),
            commit=lambda state, kind, body, found: self.commit(kind, body, found),
            emit=lambda state, name, evals, **over: self.emit(name, evals, **over),
            record=lambda *args: self.counters.__setitem__("tasks", self.counters["tasks"] + 1),
            exhausted=lambda state, task_id, kind, issues: self.emit(
                "blocker.raised", ph.E_PLAN, severity=ph.ERROR, error_code=cu.CORRECTION_EXHAUSTED))
        self.emit("task.started", ph.E_CURATOR, status=cu.TASK_KIND)
        try:
            found = cu.curate(
                runner, self.state, cu.curator_context(book, self.deps.catalog, held),
                profile=profile, facts=cu.layout_facts(self.figures), figure_data=self.figure_refs,
                coverage={row.visual_evidence_id: f"fig_{row.visual_evidence_id}"
                          for row in book.evidence},
                lineage={"parents": [held.model_dump(mode="json")], "versions": dict(book.versions),
                         "plan_id": f"fp:{self.stage_run_id}"})
        except ValueError as bad:
            self.stop(ph.FAILED, str(getattr(bad, "code", cu.SHAPE_INVALID)))
            return None
        self.counters["corrections"] = len(self.state["corrections"])
        if found.plan is None:
            self.emit("task.failed", ph.E_PLAN, severity=ph.ERROR, error_code=found.error_code)
            self.stop(str(found.status), str(found.error_code), found.detail_codes)
            return None
        self.emit("task.completed", ph.E_PLAN, status=cu.TASK_KIND)
        return found.plan

    # -- gates 3 and 4: compile then render, sequentially, in plan order ----

    def draw(self, plan: pc.FigurePlanV1,
             profile: pc.MethodProfileV1) -> tuple[pc.FigureSpecV1, ...] | None:
        # §13/§14: one specification, one SVG, one PNG, and one frozen-value table per accepted
        # figure, in plan order. Gate 3 reads the compiled document itself; gate 4 measures the
        # rendered bytes and the table. Nothing commits before the gate over it has passed.
        cat, held, figure_data = self.deps.catalog, self.held[ph.PLAN], self.figures
        entries = {row.figure_id: row for row in plan.figures}
        templates = {row.template_id: row for row in cat.templates}
        out, width = self.deps.render_root / self.stage_run_id, co.panel_width(cat.display_profile)
        try:
            self.identities["font_sha256"] = rd.register_font(cat, self.deps.repo_root)
            specs = co.compile_figures(plan, cat, profile, figure_data,
                                       {"parents": (held,), "versions": dict(plan.versions)})
        except pc.PresentationError as bad:
            self.stop(ph.FAILED, bad.code)
            return None
        for spec in specs:
            entry, template = entries[spec.figure_id], templates[spec.template_id]
            self.emit("task.started", ph.E_COMPILE, status=spec.figure_id)
            document = co.figure_document(spec, entry, template, profile, figure_data, width)
            if codes := ph.document_codes(document):
                self.stop(ph.FAILED, codes[0], codes)
                return None
            self.commit(ph.SPEC, spec.canonical_payload(), self.parent(held))
            try:
                output = rd.render(spec, document, cat, out, {
                    "parents": (self.held[ph.SPEC],), "versions": dict(spec.versions)})
            except (pc.PresentationError, OSError, ValueError) as bad:
                self.stop(ph.FAILED, str(getattr(bad, "code", rd.PARITY_BROKEN)))
                return None
            table = rd.accessible_table(spec, entry, figure_data,
                                        cat.limits.accessible_table_max_rows)
            if codes := rd.render_codes(spec, output, template, cat, table):
                self.emit("task.failed", ph.E_RENDER, severity=ph.ERROR, error_code=codes[0])
                self.stop(rd.render_status(codes), codes[0], codes)
                return None
            # §14: each figure's frozen-value table is one more committed object of its render.
            blob = json.dumps(table, sort_keys=True).encode("utf-8")
            (path := out / f"{spec.figure_id}.table.json").write_bytes(blob)
            self.commit(ph.RENDER, output.artifact.model_copy(update={
                "objects": dict(output.artifact.objects) | {"table": str(path)},
                "object_hashes": dict(output.artifact.object_hashes) | {
                    "table": hashlib.sha256(blob).hexdigest()}}).canonical_payload(),
                self.parent(self.held[ph.SPEC]))
            self.emit("task.completed", ph.E_RENDER, status=spec.figure_id)
        return specs

    # -- gate 5: the §15 summary, the final report, and the one bundle ------

    def summarise(self, book: pc.PresentationContextManifestV1, view: ArtifactRef) -> str:
        # §15 in order: question and claim status, the approved graph, the primary estimate and
        # its uncertainty, diagnostics, prespecified sensitivities, mandatory qualifications with
        # the assumption reminder, then provenance. Every line carries its own citation.
        claim = self.upstream["claim_judgment"].artifact_id
        judgment, cat = self.payload(claim), self.deps.catalog
        legend = ph.short(self.payload(view.artifact_id).get("legend_text"))
        lines = [((f"Estimand {book.approved['estimand_id']} by method"
                   f" {book.approved['method_id']} is judged {book.claim_status}."), claim),
                 (("The approved causal graph is placed unchanged at its approved hash; its"
                   f" legend reads {legend}."), view.artifact_id)]
        lines += [((f"For {row['contrast_id']} the frozen estimate is {row['estimate']}"
                    f" {row['estimate_units']}, interval {row['interval_lower']} to"
                    f" {row['interval_upper']} at {row['confidence_level']}:"
                    f" {ph.short(row['effect_statement'])}"), f"{row['contrast_id']}, {claim}")
                  for row in judgment["items"]]
        lines += [(("Identification support, every required diagnostic, and every prespecified"
                    " sensitivity are frozen, at their own terminal statuses, in the estimation"
                    " bundle."), self.upstream["estimation_bundle"].artifact_id)]
        lines += [(f"Mandatory qualification {name} stands beside the primary result.", name)
                  for name in book.qualification_ids]
        lines += [("Every statement above holds only under the approved design's assumptions.",
                   book.approved["question_id"]),
                  ((f"Rendered from catalog {cat.catalog_version} at profile"
                    f" {cat.display_profile.display_profile_id}, theme {cat.theme_id}, font"
                    f" {cat.font_id}, compiler {co.COMPILER_VERSION}, renderer"
                    f" {rd.RENDERER_VERSION}."), self.held[ph.PLAN].artifact_id)]
        return "\n".join(f"{text} [{cites}]" for text, cites in lines)

    def bundle(self, book: pc.PresentationContextManifestV1, specs: Sequence[pc.FigureSpecV1],
               view: ArtifactRef) -> ArtifactRef | None:
        # Gate 5: every substantive sentence and every bundle parent resolves to a frozen artifact
        # before the one immutable product is committed. §14's frozen-value tables are objects of
        # each render, so the bundle's accessible tables are exactly those render artifacts.
        book_ref, plan = self.held[ph.MANIFEST], self.held[ph.PLAN]
        drawn, renders = self.kept(ph.SPEC), self.kept(ph.RENDER)
        self.summary = self.summarise(book, view)
        codes = ph.summary_codes(self.summary, frozenset(book.allowlists["artifact_ids"]) | {
            *book.statement_ids, *book.qualification_ids,
            *(row.artifact_id for _, row in self.trail)}) + tuple(
            f"{ph.UNFROZEN}:{row.artifact_id}" for _, row in self.trail
            if self.deps.products.find_artifact_hash(row.artifact_id) != row.content_hash)
        report = self.commit(ph.REPORT, {
            "schema_version": "presentation-validation-report.v1", "gate": 5, "codes": list(codes),
            "stage_run_id": self.stage_run_id,
            "renderer_fingerprint": rd.fingerprint(self.deps.catalog),
            "spec_hashes": {row.figure_id: row.spec_hash() for row in specs}}, self.parent(plan))
        if codes:
            self.stop(ph.FAILED, codes[0], codes)
            return None
        self.status = ph.QUALIFIED if book.qualification_ids else ph.COMPLETE
        return ph.ref(self.commit(ph.BUNDLE, pc.PresentationBundleV1(
            parents=(book_ref, plan), versions=dict(book.versions), context_manifest=book_ref,
            plan=plan, specs=drawn, renders=renders, accessible_tables=renders,
            causal_graph_view=view, validation_report=ph.ref(report),
            summary=self.summary).canonical_payload(),
            self.parent(book_ref, plan, drawn[0], renders[0])))


def run_presentation(deps: ph.PresentationDeps, *, analysis_id: str, stage_run_id: str,
                     handoff_manifest: HandoffManifestV1, estimation_outcome: ArtifactRef,
                     approved_graph_view: ArtifactRef,
                     presentation_revision: int = 1) -> ph.PresentationRunResult:
    # One presentation revision, run straight through: entry, manifest, the one curator call,
    # compile and render in plan order, final validation, bundle. A typed status short-circuits to
    # the close, so no later gate can waive an earlier failure (§6, §16). A rerun is artifact
    # replay (D-035): a NEW stage_run_id whose deterministic ids make every recommit a no-op.
    run = PresentationCoordinator(deps, analysis_id=analysis_id, stage_run_id=stage_run_id,
                                  revision=presentation_revision)
    run.open_run()
    if (profile := run.entry(handoff_manifest, estimation_outcome, approved_graph_view)) is None:
        return run.close(None)
    book = run.freeze(profile, estimation_outcome, approved_graph_view)
    if (plan := run.curate(book, profile)) is None:
        return run.close(None)
    specs = run.draw(plan, profile)
    return run.close(None if specs is None else run.bundle(book, specs, approved_graph_view))
