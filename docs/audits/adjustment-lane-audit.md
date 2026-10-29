# Adjustment (DoWhy) lane: an audit

> Historical audit of an earlier implementation. Its module names and recommendations are not the current component specification. See the [adjustment component](../components/11-adjustment.md) and [generated ladder](../ladders/adjustment.md) for the current structure.

Date: 2026-10-05. Branch `desk-tools-2026-09-28`. Read from the code, not the docs. Paths are under `causal_agent/`;
`N` is `families/adjustment/lane/nodes.py`.

The audit has three stages and one list of findings:

1. What the lane has before it designs
2. How the design is built
3. What happens after the design
4. Findings: wrong, judged by code, out of step, duplicated

```
desk interview ─▶ memory ─▶ Handoff (handoff.json) ─▶ load ─▶ case            STAGE 1: data in
   ─▶ pair ─ mechanism ─ time ─ roles ─ post_roles ─ merge_graph ─ verify_graph
   ─ road ─ heterogeneity ─ threats ─ check_design ─(assess)─ pick_estimator  STAGE 2: design
   ─▶ freeze_design ─ analyse ×C ─ after_analyse ─ interpret ×C ─ figures
   ─ assemble ─▶ result.json ─▶ desk record ─▶ material ─▶ chat after        STAGE 3: after
any stop or ask-back ─▶ feasibility ─▶ figures ─▶ assemble
```

---

## Stage 1: what the lane has before it designs

### 1.1 The desk gathers evidence (before the hand-off)

| Step | Where | Who | What |
|---|---|---|---|
| Frame | `desk/nodes/frame.py:117` | model, gated | outcome, cause, scope, relevant columns, intent |
| Probes and data facts | `desk/nodes/shared.py:99` | code | every family's probes plus `probe:data.*`, on the whole CSV |
| Adjustment probes | `families/adjustment/probes.py:13` | code | `arms`, `overlap` |
| Family choice | `desk/nodes/decide.py:133` | code if one family fits, else one model call | `family`, `why`, `over` |
| Designer brief | `desk/designer.py:91` | model | `brief` (bets_on, road, `when` per column) |

### 1.2 The hand-off (`desk/handoff.py:39`, written at `desk/nodes/run.py:46`)

| Field | Comes from | Status kept? |
|---|---|---|
| `question` | the person's words | n/a |
| `story` | `claim:story.text`, verbatim | n/a |
| `outcome`, `scope`, `relevant_columns`, `intent` | the frame | **no**: model draft |
| `treatment` | `claim:assignment.treatment_column`, else `frame.cause` | **no** |
| `grain`, `sampling`, `missing`, `change`, `assignment`, `treated_level` | `memory.values_of` (`memory/records.py:119`) | **no**: confirmed and drafted are mixed |
| `control_level` | other level if the whole file has two | code |
| `columns` (ColumnBrief) | `memory/views.py:29`, roles from `memory/ops.py:230` | **yes**: each field carries `provenance` |
| `beliefs` | `memory/views.py:57` | **yes** |
| `unknowns`, `contradictions`, `said`, `claims` | memory | **yes** |
| `probes`, `dataset_facts` | code | data facts |
| `brief` | Designer | model draft |
| `design` (AdjustmentDesign) | `families/adjustment/handoff.py:11` | **no**: see below |

The adjustment block is built by code from those inputs:

- `adjustment_candidates`: `assignment.depends_on`, plus every column with `when == "before"` and not moved by the change.
- `forbidden`: every column with `when` at or after the change, or moved by it.
- `instrument`, `mediator`, `unobserved_confounding`: from beliefs where `Belief.known()` holds, and `known()` accepts
  `drafted` (`common/contracts/pack.py:232`).
- `voluntary_uptake`: from `assignment.kind`, else `assignment.movable`.

### 1.3 Inside the lane, before the first rung

| Node | Who | Reads | Writes | Stops when |
|---|---|---|---|---|
| `load` (`N:121`) + `lane/intake.py:294` | code | handoff, CSV | `run_dir`, `table.csv`, `columns`, `outcome_kind`, `target_units`, `treatment_levels`, empty `Ladder` | intent is not an effect; no treatment; outcome or treatment missing; outcome neither numeric nor two-valued; unknown target; numeric treatment with more than 12 levels |
| `case` (`lane/case.py:208`) | code | handoff, `beliefs.yaml` | `case`: each column field and belief sorted into fact / draft / contested / open, plus flags | never |

`load` keeps only the treatment, the outcome, `relevant_columns` and the block's columns, applies the scope filter, then
drops every row with a gap. Forbidden columns are never loaded.

The lane never reads `family.yaml`. Its `assumes` reaches the lane only as the `chosen_assumption` string.

---

## Stage 2: how the design is built

Two retry mechanisms apply to the model steps. `run_episode` (`lane/episode.py:126`) is a look phase with read-only tools
within a budget (`checks.yaml:38`), then one typed answer, then the gate, up to 3 tries. A LangGraph `RetryPolicy` (3
attempts) also wraps the node.

| # | Rung | Who decides | Gate rejects |
|---|---|---|---|
| 0 | `pair` (`N:255`) | code if `treated_level` is seen and a control is found; else episode (budget 2) | no contrast, unseen level, control = treated, duplicate |
| 1 | `mechanism` (`N:326`) | code if `kind == "lottery"` or drivers are named; else episode (budget 4); `kind` always from the pack | driver not a column, offer/uptake not a column or the same, self-selection under a lottery, no cites, cite that does not resolve |
| 2 | `time` (`N:404`) | code: fact, else **draft**, else brief | nothing |
| 3 | `roles` (`N:567`) | code for columns the pack settles (`fact_relation`); episode (budget 8) for the rest | column missing or repeated; claim without a cited reason; cite unresolved; feeds and is moved by the treatment; contradicts a settled fact; departs from the last reading without a named `Departure`; redundancy or nesting link without a backing fact |
| 4 | `post_roles` (`N:629`) | same split, episode (budget 4) | presence, no cites, the same relation checks |
| 5 | `merge_graph` (`N:691`) | code | nothing; builds the DAG, adds `HIDDEN`, drops forbidden, redundant and nested columns, sets the mediator path, applies revisions |
| 5 | `verify_graph` (`N:761`) | code | cycle, a node that is not a column, a column with no role → stop |
| 6 | `road` (`N:811`) | DoWhy identifies; code if one road or the brief names one; asks back when hidden and nothing identifies; episode (no tools) if several roads | road not open, empty `why`, cite unresolved |
| 7 | `heterogeneity` (`N:958`) | episode over the candidates; target by code | too many, not a candidate, twice, no cite, cite unresolved |
| 8 | `threats` (`N:1022`) | code: `pack_threats` (selection, outcome timing, missing data) plus compliance and self-selection | nothing |
| 9 | `check_design` (`N:1055`) | code: identification, arms, overlap, balance, outside-candidates, case flags, threats and unsure items as flags | nothing; unsure on a catalogue claim adds a `needs.unsettled` decline |
| 10 | `assess` (`N:1096`) | `decide_by_code` first (stop / ask / clear), then the model in a **hand-written loop** | proceed under a hard flag; revise with no revision or on an unflagged column; cite unresolved. Revise goes back to `merge_graph`, at most 3 times |
| 11 | `pick_estimator` (`N:1216`) | catalogue filtered by code, model picks in a **hand-written loop** | name not offered, cite unresolved (bad cites are first dropped silently) |

---

## Stage 3: what happens after the design

| Step | Who | What | Writes |
|---|---|---|---|
| `freeze_design` (`N:1269`) | code | if the estimator's road differs, rewrites `est.kind`; refuters are every yaml entry that applies; `also_run` if it applies | `run_dir/design.json`, `design.md` |
| `analyse` × contrast (`N:1307`) | code (DoWhy) | estimate; refuters if it fit; secondary always; per-modifier fits if it fit | state only |
| `after_analyse` (`N:1387`) | code | on a failed fit, re-pick once with the estimator excluded; else stop | |
| `interpret` × contrast (`N:1505`) | model, **hand-written loop** | must cite every flag, the CI and the modifier values; `effect_stated` within `effect_tolerance` (1%) | **on failure the answer is kept** and `interpret_errors` set |
| `figures` (`N:1544`) | code | graph, balance, effect and refutations, effect by modifier; each spec checked against `LF.ok_addresses` | `run_dir/figures.json` |
| `assemble` (`N:1566`) | code | report and result | `report.md`, `artifacts.json`, result |
| desk `pipeline` (`desk/pipeline.py:39`) | code | subprocess; `record()` takes the first non-secondary, error-free estimate | `designs/<n>/result.json`, `record.json`, `figures.json` |
| `desk/material.py:103` | code | rebuilds every address from the record | |
| `desk/nodes/after.py:144` + `desk/explainer.py:123` | model, gated | cites must be in the material; every number within 1% of a material number or present in its text | |

---

## Findings

Severity: **A** gives a wrong or misleading result. **B** means code makes a judgement. **C** means two parts disagree.
**D** is duplicated code.

### A. Wrong or misleading

| # | Finding | Where |
|---|---|---|
| A1 | A model-drafted claim reaches the lane as if the person had said it. `values_of` drops status, so `h.assignment`, `change`, `sampling` and the rest mix confirmed and drafted values. The mechanism rung then labels that value `by="pack"`. | `memory/records.py:119`, `desk/handoff.py:54`, `N:336` |
| A2 | A drafted belief sets the design. `Belief.known()` accepts `drafted`, so a model's guess of an instrument, mediator or hidden confounder fixes the graph and the road. This breaks "a belief is written only on the person's word". | `pack.py:232`, `families/adjustment/handoff.py:16-26`, `N:429-434`, `N:702` |
| A3 | A drafted `when` excludes a column outright. The block is blind to status, so `forbidden` takes drafts, while `case` treats the same field as a draft. | `families/adjustment/handoff.py:14-15` against `N:445-448`; `N:718` |
| A4 | A settled cite names a field that does not exist: `col:<k>.moved` instead of `col:<k>.moved_by_change`. | `N:446` |
| A5 | An interpretation that failed its gate is kept. `interpret_errors` never reaches `result.json`, so the desk shows it unmarked, and the chat after counts any number in its text as grounded. | `N:1535`, `lane/records.py:56-76`, `desk/explainer.py:118`, `desk/material.py:194` |
| A6 | `freeze_design` may switch the road (backdoor → frontdoor or IV) after the checks ran on the backdoor set. The ladder still says the old road, and the balance figure and the interpret material still show the old adjustment set. | `N:1272`, `N:1418`, `N:1453` |
| A7 | With several contrasts, only the first gets an effect figure, and the desk labels `run.effect` as the first contrast. | `N:1557`, `desk/material.py:279` |
| A8 | `assess` can revise only for balance, overlap, separation or identification. A threat or unsure flag adds no column, so a revision that answers one always fails the gate. | `N:1116-1118` |

### B. Code making a judgement

| # | Default guess | Where |
|---|---|---|
| B1 | `self_selection = kind == "own_choice"` when uptake is unset | `N:342` |
| B2 | A column the decision looked at, fixed before the change, is set to `affects_outcome=True`. The pack says it fed the treatment, not that it moves the outcome. | `N:471-482` |
| B3 | An asked "average" becomes "on the treated" whenever the mechanism says self-selection | `N:947-955` |
| B4 | The timing rung uses a draft `when` as if it were settled | `N:411` |
| B5 | Road's code-written `why` says the person denied an instrument and a mediator, but the branch is also reached on `unknown` or `contradiction` | `N:872` |
| B6 | `PostRole.relation()` fixes `affects_treatment=False` | `contracts.py:157` |
| B7 | The model can never set `mechanism.kind`; code overwrites it with the pack's value, even when that is empty | `N:388` |

### C. Two parts that disagree

| # | Finding | Where |
|---|---|---|
| C1 | `roles` stores records with settled claims applied; `post_roles` stores the raw record. Ladder lines can show what the pack contradicts. | `N:624` vs `N:671` |
| C2 | A column's timing is read three ways: fact→draft→brief, fact→brief, brief only | `N:411`, `lane/ladder.py:133`, `lane/nodes.py:89` |
| C3 | Treatment levels: whole file at the desk, filtered table in the lane, so `control_level` can disagree | `desk/handoff.py:76`, `N:167`, `N:273` |
| C4 | Treated mask with and without `strip()` | `memory/checks.py:141`, `N:220` |
| C5 | `PostRoles.unsure` is never gated; the other rungs gate theirs | `contracts.py` |
| C6 | Cites are matched exactly in interpret and the desk gate, but normalised in pick and figures | `N:1524`, `explainer.py:135`, `N:1255`, `lane/figures.py:29` |
| C7 | The address list is built three times with different suffixes; a failed estimate is `{tag}.error` in the lane and `<c>.<method>.error` in the desk; no one addresses the sensitivity range the figure draws | `N:1416`, `lane/figures.py:62`, `desk/material.py:158` |
| C8 | "Primary estimate" is chosen four ways | `N:1379`, `viz/postviz/common.py:11`, `desk/pipeline.py:70`, `common.py:83` |
| C9 | The refutation prefix is chosen by `specialist == "dowhy"` in the desk, which names an engine in the core, and by the registry elsewhere | `desk/material.py:180,315` vs `desk/nodes/run.py:138` |
| C10 | The arms floor (20) is declared twice and kept in step only by a comment | `memory/checks.yaml:12`, `lane/knowledge/checks.yaml:6` |

### D. Duplicated code

**Inside the adjustment lane and the desk**

| # | Duplicate | Where |
|---|---|---|
| D1 | Seven ways to resolve a cite: `V.cites_resolve`, `V.resolves`, `_resolver`, `_made_here`, bare `h.resolve` (assess revisions), the exact `allowed` set (interpret), `LINK_CITES` substring | `lane/verify.py:18,29`, `N:223,1406,1143,1524,585` |
| D2 | Three retry layers with three constants: `MAX_TRIES`, `MAX_MODEL_RETRIES`, `RetryPolicy`, plus `after.py`'s own | `episode.py:26`, `lane/nodes.py:22`, `common/llm.py:25`, `after.py:28` |
| D3 | The 1% tolerance in yaml (lane) and hard-coded (desk) | `checks.yaml:25`, `explainer.py:25` |
| D4 | The treatment rule (`treatment_column or frame.cause`) in three places | `desk/handoff.py:56`, `desk/nodes/shared.py:107`, `memory/ops.py:245` |
| D5 | Target units derived three times | `families/adjustment/handoff.py:28`, `N:148`, `N:950` |
| D6 | Voluntary uptake derived twice | `families/adjustment/handoff.py:20`, `N:343` |
| D7 | Dataset entry and CSV resolved twice | `desk/handoff.py:53`, `lane/intake.py:296` |
| D8 | Material text built twice, in the lane and in the desk | `N:1444`, `desk/material.py:103` |
| D9 | Figures drawn twice: in the lane, and by the desk fallback, which skips the address check | `N:1557`, `desk/nodes/run.py:137` |
| D10 | "null"/"none" blank tests in four places, besides `intake.blank` | `N:235,356,588`, `lane/intake.py:99` |
| D11 | `TREATED = "treated"` twice | `adapter.py:29`, `checks.py:12` |
| D12 | `_slug` copied | `desk/material.py:50`, `common/contracts/base.py:45` |
| D13 | Scope string rebuilt word for word | `N:284`, `lane/nodes.py:63` |
| D14 | `_thought` defined and never called; `interpret_attempts` set and never read | `N:113`, `N:195` |

**Across the three lanes (plumbing that belongs in `lane/`)**

| # | Duplicate | Copies |
|---|---|---|
| D15 | `_ladder`, `_budget`, `_resolver`, `_made_here`, `_episodes`, `_presence_errors`, identical | 3 each |
| D16 | The nine-name alias block `_writer, _question, ... = L.writer, ...`, byte-identical | 3 |
| D17 | Hand-written `for _ in range(MAX_MODEL_RETRIES)` loops in `assess`, `pick_estimator`, `interpret`, where `run_episode(tools=None)` already does the job | 9 (17 nodes already use `run_episode`) |
| D18 | `pick_estimator` loop and gate near-identical | 3 |
| D19 | `assess` opening (`decide_by_code` → stop/ask/clear) identical; the A and D revise loops about 80% the same | 3 |
| D20 | `interpret` gate identical (allowed, required, tolerance) | 3 |
| D21 | `heterogeneity` node almost the same (D and R 99%) | 3 |
| D22 | `assemble`: header, ladder, episodes, stop, thoughts and result blocks identical | 3 |
| D23 | `_material` / `_addresses` share a block of beliefs, checks, declines, ladder and facts | 3 |
| D24 | `load` intake-stop, target-units stop and update keys | 3 |
| D25 | `check_design` and `threats` endings identical | 3 |
| D26 | Re-pick on a failed fit: `after_analyse` in A, inline in D and R, same stop text | 3 |
| D27 | `SpecialistState` repeats ten fields and two reducers | 3 |

---

## Suggested order of work

Each step is one commit.

1. **Status reaches the lane.** The hand-off carries provenance for claim fields as it does for columns. `Belief.known()`
   means confirmed only. The block builder and `timing` read status, and a draft is a draft everywhere. (A1–A3, B4, C2)
2. **Fix `moved` → `moved_by_change`.** (A4)
3. **A failed interpretation is marked.** `interpret_errors` goes into the result; the desk shows it and does not ground
   numbers on it. (A5)
4. **One road.** `freeze_design` cannot change the road, or a change re-runs the checks and updates the ladder. (A6)
5. **One cite resolver, one address list, one tolerance, one primary.** These live in `lane/` and the desk imports
   them. (C6–C9, D1, D3, D8)
6. **Every model step through `run_episode`.** Delete the hand-written loops and the extra constants. (D2, D17–D20)
7. **Move the copied helpers and node endings into `lane/`.** (D15, D16, D21–D27)
8. **Each code default (B1–B3, B5–B7) becomes either a fact with an address or a judgement.**
9. **Small cleanups.** (D4–D7, D10–D14, C3–C5, C10)
