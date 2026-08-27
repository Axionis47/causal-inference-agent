# T-030 — Presentation substrate: contracts, visualization catalog, context manifest

Status: frozen for implementation
Owning PRD: PRD-005 §5, §7.1, §8, §10, §18
Depends on: T-026 (estimation handoff shape), T-029

## 1. Deliverables

1. `src/causal/presentation/contracts.py` (≤ 240 logical) — strict pydantic:
   `VisualizationCatalogV1` (§8: profiles, templates with FINITE panel/series/label/
   annotation/height bounds — absent or unbounded capacity fails the loader; display
   profile `desktop-736-v1`; theme; font id + SHA-256; catalog-wide limits ≤6
   figures/≤3 panels), `PresentationContextManifestV1` (§7.1),
   `PresentationCuratorContextV1` + `FigurePlanDraftV1` + `FigurePlanV1` (§10),
   `FigureSpecV1` (§13 fields; canonical hash = identity), `RenderArtifactV1`,
   `PresentationBundleV1` + `PresentationOutcomeV1` (§1 closed status set),
   `PresentationRunV1` (§18), one `PresentationError`.
2. `src/causal/presentation/catalog.py` (≤ 120 logical) — fail-closed loader for
   `registries/visualization-catalog.v1.json` + typed accessors (profile for method,
   compatible templates per evidence id, capacity checks) + the §5 thirteen-condition
   entry gate feeding gate 1 (`entry_codes` idiom from estimation/plancompile.py).
3. `registries/visualization-catalog.v1.json` (≤ 200 non-empty — declarative is
   TIGHT: ~305 left for all of Wave 2) — 4 method profiles per §12's evidence
   questions; a MINIMAL honest template set (point-and-interval effect template;
   grouped-bar counts template; event-time/series template; binned-scatter+fit
   template; balance dot-plot template — reuse across methods via
   allowed-profile lists rather than per-method duplicates); exact §13 display
   profile; theme; font row (id `noto-sans-v2.015-variable-normal`, the SC-pinned
   SHA-256).
4. `registries/artifact-types.v1.json` — append the presentation rows (bundle, plan,
   spec, renders, manifest; producer presentation-coordinator; keep rows minimal).
5. Font vendoring attempt: fetch the pinned Noto Sans file per SC's source
   commit into `assets/fonts/` and verify the SHA-256. Vendored assets are
   byte-counted, not line-counted. If the network refuses, record the absence — the
   renderer must raise the §13 blocker and its tests skip per the D-042 idiom.

## 2. Tests (≤ 250 logical)

Round-trips + rejections (unbounded template capacity, unknown profile, >6 figures);
catalog loader fail-closed; entry gate: golden path from a T-026-shaped estimation
handoff fixture + each §5 condition violated → stable code; spec-hash determinism.

## 3. Constraints

presentation ≤ 360 this task (target total ≈1,100/1,500); declarative ≤ +210;
tests ≤ 250; modules +2 (after T-029 count). No imports from causal.estimation
internals beyond contracts (read the handoff artifacts through shared readers).
Wave 2 shares ONE rethink. Committed code wins; record deviations.
