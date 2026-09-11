# Known-dataset live checkpoint

Checkpoint recorded: 2026-09-09 01:02:40 UTC

This is an execution record, not a release approval. A cell is delivered only when real Vertex
model decisions reached a terminal application outcome. Scripted runs and evaluator-generated
`unknown` answers do not count as human-in-the-loop coverage. Results from different code
revisions are kept separate and are not combined into an implied clean gate.

| Method | Known dataset | Rich context | Sparse context |
|---|---|---|---|
| Randomized | Lalonde/NSW | Delivered live output: `complete`, zero hard failures. Report `known-finalprobe-lalonde_nsw_rct-4b810b38ad6a416abbc0a287bf07d2df.json`. Scientific status remains conditional on documenting that `treat` is assignment rather than receipt. | Not run. |
| Randomized | Resume audit | Not run. Scientifically valid point contrast, but current projection lacks employment-ad cluster identity, so inference is incomplete. | Not run. |
| AIPW | Groupon | Legitimate `needs_context` after asking what the undocumented treatment label means. Report `known-rerun-groupon_aipw-190a18eeb6874b92a5f58512793b985b.json`. Not a positive causal fixture without provenance for exposure construction and covariate timing. | Not run. |
| AIPW | NHEFS | Not run. Independent review rates it the strongest known AIPW fixture. | Not run. |
| DiD | Federal minimum wage | Software failure in cyclic causal graph correction. Report `known-high-minimum_wage_did-640ba818c07e4fcca474ca54fe549e7e.json`. Separately, the fixture's single 2009 adoption rule conflicts with the staged 2007–09 federal increases. | Not run. |
| DiD | Card–Krueger | Not run. Canonical two-period DiD, but not a positive case for the current pack's two-pre-period minimum; the transform also performs post-treatment complete-case selection. | Not run. |
| Sharp RDD | Senate | Failed current probes. The stale-question repair removed six false timing asks; the next run reached presentation and returned `needs_template`, and a later run failed bounded role-ledger correction. Reports `known-rerun-rd_senate_sharp-5b88536a11db410caf2d5caaf0a65d7a.json` and `known-finalprobe-rd_senate_sharp-6d58c4bfbce24813811658b033860bfa.json`. | Two probes failed before a production clarification interrupt, first in method binding and then role evidence. The final corrected probe process expired without returning an interrupt or terminal result. No isolated human answer or approval completed. |
| Sharp RDD | Head Start | Not run. Scientifically suitable only for the reduced-form local effect of eligibility for OEO grant-writing assistance, not funding, participation, or service receipt. | Not run. |

## Verified repairs

- Rejected model drafts can no longer persist clarification requirements before validation.
- An explicit `unknown` answer ends the unresolved design instead of repeating the same packet.
- RCT attrition figure data now retain denominators required by the visualization contract.
- The curator receives the registered profile's evidence order, primary evidence ids, mandatory
  encodings, permitted combinations, and qualification placement.
- Graph-cycle corrections now name the live edge endpoints instead of returning only a code.
- A prerequisite gate was proven to stop method calls while an earlier requirement is open, but
  the sparse probes show that prerequisite handling must be unified before every dependent model
  boundary rather than added node by node.

Focused checks passed: 44 ask-gate/design tests before the prerequisite change; 6 targeted
ask-gate graph tests after it; 73 presentation/design-validator tests; the RCT attrition
coordinator test; and Ruff on the changed repair files. The full suite, typing, final live matrix,
and release gate have not run. The budget gate currently fails: production 16,541/16,500,
design 4,310/4,300, tests 13,515/13,500. Cleanup is still required.

## Current conclusion

One rich-context journey produced a real rendered delivery. The 16-cell target is not complete,
and genuine isolated-human HITL coverage is still zero. The principal remaining software issue
is fragmented prerequisite and revision ownership across model boundaries. The principal data
issue is that three proposed positive fixtures cannot honestly support their requested causal
interpretation under the current documentation or method contract.
