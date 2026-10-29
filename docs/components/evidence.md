# Evidence, discrepancies and verification

[Component map](README.md) · [Source ownership](source-index.md)

## Inspection baseline

Inspected the working tree on 21 October 2026 against base commit `3358e44`. Existing edits, deletions and untracked files were present before handbook work. They included diagnostic coding, trace/sandbox changes, family declarations and tests, and removal of generated evaluation datasets/reports. The handbook reflects those files as found; it does not attribute their implementation or historical test results to this documentation work.

No production source, public API, fixture dataset, runtime record, schema or generated ladder is intentionally changed by this pass. No model call, external trace lookup, live API conversation, source formatter or dataset deletion was performed. Repository content, not an external framework description, is the authority for this map.

## Evidence labels

| Label | What it establishes | What it does not establish |
|---|---|---|
| **Code-verified** | Definition, caller/consumer, gate or write path inspected in the current tree | Successful behavior in every environment |
| **Test-supported** | Existing tests assert the described invariant/scenario; relevant tests are linked | Those tests were freshly executed, or their scripted model behaved like Gemini |
| **Runtime-observed (retained report)** | A local acceptance/readback document reports a concrete real-model run and its limits | A new observation in this pass or verification of its remote trace |
| **Unresolved** | The implementation or available evidence does not support a stronger guarantee | A confirmed defect without further investigation |

Worked examples describe inspected control flow. They do not fabricate numerical results or claim new model observations. Source listings establish ownership coverage, not line-by-line auditing of every estimator implementation.

## Retained runtime evidence

| Record | Supported observation | Limit |
|---|---|---|
| [Paired DiD readback](../acceptance/readbacks/did-paired.md) | Paired analysis and two-cluster inference limitation | One sourced scenario |
| [Cigar DiD readback](../acceptance/readbacks/did-cigar.md) | Explicit stop at pre-treatment checks | Does not validate downstream estimates for that stopped run |
| [Adjustment selection](../acceptance/code-assisted-estimator-selection.md) | Conditional students design; repaired numeric evidence and estimator choice | Selector-level validation, not full causal identification or API acceptance |
| [DiD/RD diagnostic coding](../acceptance/lane-diagnostic-code.md) | Restricted diagnostic requests, citation repair and ordinary paired/sharp runs | Selected paths and platform coverage only |
| [Acceptance policy](../acceptance/context-lane-policy.md) | Defines required evidence and retained coverage | Full endpoint coverage remains to be established |

Historical suite counts in those reports are not current verification results. Removed synthetic-data reports are not cited as present acceptance evidence.

## Documentation discrepancies reconciled

| Existing description | Current code evidence | Documentation action |
|---|---|---|
| Shared lane README names `case.py`, `verify.py`, `asks.py` and a previous beliefs/checking shape | Those modules are absent; current runtime has endpoints, ledger, episode and family-owned gates | Replaced the package overview with actual modules and linked component 10 |
| Exploration README permits withholding outcome or treatment | `project.choose_drop` specifically withholds a known outcome before freeze | Corrected the README and explained the projection in component 14 |
| Exploration reference says artifacts are never read back by code | `memory.artifacts`, desk material and server projections load them | Corrected the reference and documented producers/readers |
| Exploration reference says the Reader reads every pre-run message | `desk.turns` and `interview.infer` can dispatch confident classified turns first | Corrected the request-detection description |
| API reference embeds an obsolete `SessionView` line number | The type is in `server/models.py`, at a different line | Kept a stable file link without a stale line label |
| High-level summaries omit the exploration import layer or imply the desk is the only graph | `pyproject.toml` includes explore; each built family compiles a graph | Handbook gives the actual import order and distinguishes conversation graph from specialist graphs |
| Generic family layout suggests every family must have `postviz.py` | Discontinuity generates figures in its lane artifacts module | Component 13 names its actual figure owner |

The historical [adjustment audit](../audits/adjustment-lane-audit.md) describes an earlier implementation and references removed modules. It is explicitly labeled historical. Archived ADRs remain historical references. This handbook does not revive removed shapes to satisfy those documents.

## Implementation facts that need careful wording

- `store.migrate` currently seeds memory from registered dataset/profile/claims inputs. The name does not establish old-checkpoint or old-analysis compatibility.
- `Memory.set` is a low-level primitive also used by seeding and checks. `ops.apply` is the semantic update gate, not literally the only assignment statement anywhere in the memory package.
- User exploration and family diagnostics share a Docker-only runner. Their data scopes remain different: user exploration receives the complete selected CSV, while each family constructs its restricted diagnostic input.
- The context/result envelopes retain some dictionaries. A complete typed common result schema should not be inferred from the presence of `RunRecord`.
- Analysis memory saves and multi-store deletion are not one transaction. Analysis cleanup discovers recorded runs, catches checkpoint-deletion exceptions and does not sweep every orphan/cache. Ingestion uses its own PostgreSQL publication transaction, private S3 verification and fenced cleanup.
- Discontinuity's forbidden-import entries include engine module names that do not exist in the current layout; actual fitting code is in `engine/fit.py` and `engine/postchecks.py`. This is a boundary-enforcement uncertainty recorded for later organization work. No lint rule or implementation was changed here.
- Prefreeze free-form projection withholds the known outcome. If no outcome has been identified, that protection cannot identify a column to remove. The documented behavior follows the actual caller/projection contract.

## Handbook verification

Validation checks local Markdown links and headings, the nine-section structure of all 16 chapters, unique and complete source ownership, referenced Python symbols, the documented endpoint/rung inventories, and documentation-only changes against a pre-edit file-hash baseline. It also checks whitespace and avoids em dashes in newly authored handbook prose.

The Python checker uses the existing environment through `uv run --no-sync --offline` with a temporary cache directory. Default uv cache access was unavailable in the sandbox; attempting offline synchronization lacked the build dependency cache. Neither required changing project dependencies. No production tests or model evaluations are needed to establish the new Markdown links and inventory, and none is reported as freshly passing.

Results for this documentation pass:

- 16 component chapters passed the same nine-section structure check; the handbook contains 20 Markdown pages including the map, terms guide, ownership index and this evidence record.
- All 830 local handbook links and 70 local links in the seven edited existing documents resolved.
- All 425 indexed source/support files had one primary owner; no Python/frontend production file in the inventory scope was unassigned.
- All 58 explicitly checked Python boundary symbols existed. YAML inspection confirmed 13 adjustment endpoint declarations (12 executable plus an identification diagnostic), six DiD endpoints and six discontinuity endpoints, with 14, 12 and 15 rungs respectively.
- The three Mermaid diagrams were manually cross-checked against route, graph, process-launch and path definitions. Diagram rendering was not exercised in a browser.
- Code fences, new-prose punctuation and `git diff --check` passed. File hashes confirmed that this pass changed only Markdown among pre-existing inspected files.
- No application test suite or new Gemini run was executed. Existing runtime behavior and existing working changes were preserved.

The terminology revision adds plain-language definitions with examples, links them from every component, and expands the shared-runtime walkthrough. It also explains memory, session and handoff terms where they first become important. The same structure, link, symbol, ownership and documentation-only checks passed after that revision.
