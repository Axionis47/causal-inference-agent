# T-033 — AIPW/DiD presentation-contract alignment

Status: frozen for implementation
Owning PRDs: PRD-004 §17; PRD-005 §5, §11.4, §12, §16
Depends on: T-029, T-032

## 1. Finding

The four-pack run proves AIPW and DiD estimation complete and recover the synthetic effects, but
PRD-005 blocks both with `missing_figure_data`. Their design and presentation registry rows name
`qualifications` as required visual evidence even though PRD-004 §17 and PRD-005 §12 define no
qualifications FigureData family. Qualifications are ClaimJudgment references governed separately
by PRD-005 §11.4 and gate 2.

## 2. Deliverables

1. Remove `qualifications` from the AIPW and DiD `required_visual_evidence_ids` in
   `registries/method-packs.v1.json`.
2. Move `qualifications` from required to conditional evidence in those two profiles in
   `registries/visualization-catalog.v1.json`. Keep its template compatibility, evidence order,
   and placement rule so a FigureData family can be added later without changing the profile.
3. Do not add a synthetic qualifications FigureData builder and do not relax qualification
   placement: every actual ClaimJudgment qualification must remain beside the primary result and
   in its caption under the profile rule.

## 3. Tests and verification

1. One cross-registry test proves every method pack's required visual-evidence IDs equal its
   presentation profile's required IDs and are all produced by registered estimator builders.
2. Real preparation → estimation → presentation tests prove AIPW delivers
   `complete_with_qualifications` and DiD delivers `complete` through one curator call.
3. Run the four-pack estimation journey, focused presentation suite, budget checker, ruff, mypy,
   and full pytest suite.

## 4. Constraints

Production code ≤ +8 logical; tests ≤ +60 logical; declarative change net non-positive; no new
module. The V1 files are corrected in place before external release; affected local artifacts are
not replay-compatible and must be rerun. Record measured results in D-107/D-108.

## Amendment 1 — DiD grouped support encoding

After the entry correction, DiD reached gate 4 and exposed a second defect: support rows for ten
event periods all used series `support` and category `cohort`, while the grouped-bar compiler did
not apply its documented group offset. The repeated rows stacked beyond the frozen y-domain and
produced a 1,332-pixel canvas. The DiD builder must encode cohort as series and event time as
category, and the compiler must apply an x-offset for nominal grouped bars. These are presentation
encoding corrections over already-frozen counts; no estimate, diagnostic, or domain is changed.
