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

No production-code change; tests ≤ +60 logical; declarative change net non-positive; no new
module. The V1 files are corrected in place before external release; affected local artifacts are
not replay-compatible and must be rerun. Record measured results in D-107.
