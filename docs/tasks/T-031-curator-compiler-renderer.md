# T-031 — Presentation pipeline: curator, plan validator, compiler, renderer

Status: frozen for implementation
Owning PRD: PRD-005 §9, §10, §11, §13, §14
Depends on: T-030

## 1. Deliverables

1. `src/causal/presentation/curate.py` (≤ 260 logical):
   - context builder: `PresentationCuratorContextV1` from the frozen manifest —
     approved claims/qualifications, figure-data schemas + safe bounded facts,
     compatible catalog entries, display constraints; structurally NO raw rows,
     figure payload arrays, or estimation internals.
   - the ONE curator call through shared `TaskRunner` (tool allowlist =
     `resolve_registered_layout_facts` only, max one call — implement the tool over
     frozen figure-data summaries; one initial response + ≤2 targeted corrections
     for the same stable code; typed-inability path → `needs_template`).
   - deterministic plan validator (gate 2 + §9.2 forbidden list): required-evidence
     coverage, template compatibility, quantity/unit/denominator compatibility,
     capacity bounds, qualification placement, accessibility fields present;
     stable codes; repeated failure → `needs_template` / `needs_layout_revision`.
2. `src/causal/presentation/compile.py` (≤ 190 logical): accepted `FigurePlanV1` +
   frozen figure data + catalog → one `FigureSpecV1` per figure via Altair
   (declarative construction only; §11 honesty rules mechanical: domains include
   marks + intervals + references, magnitude bars at zero, no dual axes, no
   transforms beyond metadata-permitted; missing/suppressed distinct). Spec hash
   canonical and environment-exact.
3. `src/causal/presentation/render.py` (≤ 220 logical): `vl-convert` SVG + PNG from
   the same spec at `desktop-736-v1` (736 logical px, 24 padding, 1472 PNG, pinned
   theme + vendored font — missing/mismatched font raises the §13 blocker, no
   substitution); accessible description + frozen-value table per figure (§14);
   render/accessibility validator (gate 4: legible, unclipped, SVG/PNG parity,
   description accuracy) + renderer fingerprint record (§13.1).
4. `prompts/presentation/curator.v1.txt` (~50 declarative) — hydrated closed
   vocabularies (template ids, evidence ids, statuses, artifact ids), §9.1/§9.2
   may/may-not lists, cite-or-omit discipline.
5. Pins: `uv add altair==6.2.2 vl-convert-python==1.9.0.post1` if absent from the
   lock (check first); commit lock delta with the slice.

## 2. Tests (≤ 350 logical)

Fake-gateway curator: golden plan accepted; hidden-evidence draft rejected with
stable code → corrected; unregistered template refused; typed inability →
needs_template; layout-fact tool: one call allowed, second denied. Compiler: every
template family compiles from fixture figure data; identical inputs → identical spec
hash; forbidden construct (transform/remote asset) impossible by construction
(assert spec fields). Renderer: SVG+PNG produced for short/long/sparse/dense label
fixtures in the one profile; clipping fixture → needs_layout_revision; font-absent →
blocker (skip render-parity when the vendored font is missing per D-042);
frozen-value table matches spec data exactly.

## 3. Constraints

presentation ≤ 1,030 total after (+670); declarative ≤ +55; tests ≤ 350; modules +3.
Curator output NEVER reaches the compiler uncommitted (plan validator + commit
between). No matplotlib, no vega-embed, no new chart framework. Wave 2 shares one
rethink. Committed code wins; record deviations.
