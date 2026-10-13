# 10. A sealed rung is replayed, not rerun

Proposed 2026-10-05, after [the adjustment lane audit](../audits/adjustment-lane-audit.md). Nothing in it is built yet.
ADR 0008 and 0009 stand. This one says what the pack must carry for the adjustment lane to reason, in what order the
reasoning runs, what each rung is allowed to read, and what is kept between runs so no rung is reasoned twice while its
inputs stand.

## The problem it answers

Two runs on the same pack can end differently, and a run that stops at one rung throws away every rung below it.

The run under `.artifacts/runs/students-dowhy-9baefe8d` stopped at the mechanism rung after three refusals for "no
citations given", with no tool call made. The case already held `col:lunch.feeds_treatment = true` and
`col:parental_level_of_education.feeds_treatment = true` as confirmed facts. The rung reads its drivers from
`claim:assignment.depends_on` alone ([nodes.py:326](../../causal_agent/families/adjustment/lane/nodes.py)), so the
same fact at another address did not settle it, and the model was asked to judge what the person had already said. The
refusal named no address the model could have cited. The pair rung, already climbed, was lost with the stop.

The lane is compiled with no checkpointer ([lane/graph.py:17](../../causal_agent/lane/graph.py)) and writes its
ladder only at `assemble`. The next run starts at `load` again. An ask-back from the road rung, the one question a lane
may ask, reruns the five rungs below it with the model free to answer each differently. That is the circle.

The pack carries one fact in up to four shapes: the column brief with provenance, the `assignment` dict without it, the
`claims` snapshot, and the family block. The audit's A1 to A3 are that drift. A lane cannot replay what it cannot
compare, and it cannot compare a fact that has four spellings.

## The decision

**One fact, one address, with its status.** The pack carries every claim the lane reads as one addressed entry: the
value, the status, the source, the sentence it rests on. The bare dicts, the claims snapshot, and any block field that
does not name the addresses it rests on go. A `confirmed` fact is the person's word or a data check and a rung takes
it. A `drafted` one is the last reading; a rung may depart from it with a cited reason. An `unknown`, `refuted` or
`contradiction` one is a flag. Code never fills a claim that has no address: no default for a missing `moves_outcome`,
no target switched by code, no draft read as settled (audit B2, B3, B4).

**A rung declares what it reads and is sealed when its gate passes.** Each rung lists the addresses it reads. The
harness takes the rung's inputs from the pack by that list, and a test proves the rung reads nothing else. When the
gate accepts a record, the harness writes the record, the episode log and the inputs to the run folder at once, as a
seal. A stop seals nothing for the failing rung and leaves every seal below it in place.

**A sealed rung is replayed when its inputs stand.** The desk puts the previous design's seals into the next pack as
`prior`. The lane climbs in order. A rung whose every input has the same value and status as its seal, and whose rungs
below were replayed or computed to the same record, is taken from the seal with `by = "replay"` and no model call. The
first rung whose inputs moved runs, and every rung above it runs. After the freeze nothing is replayed: the design is
the contract, and the engine runs on it.

## The pack the adjustment lane needs

Every address below is one the grammar in [pack-and-addresses.md](../pack-and-addresses.md) already names. `facts` is
the only place a claim's value lives. `columns` carries the file's facts and nothing the person said. `block` fields
carry the addresses they rest on and are filled from confirmed facts only; a block field whose evidence is a draft is
null, and the rung that needs it sees the draft as the last reading. `prior` is the previous design's seals, or null.

```json
{
  "pack_name": "students",
  "design_id": 2,
  "memory_version": 63,
  "csv": "data/raw/students.csv",
  "family": "adjustment",
  "specialist": "dowhy",
  "question": "Did completing the prep course raise math scores?",
  "intent": "effect_of_change",
  "scope": {"population_filter": null, "window": null, "contrast": "switch", "target": "average"},
  "pair": {
    "outcome": "math_score",
    "treatment": "test_preparation_course",
    "treated_level": "completed",
    "control_level": "none",
    "rests_on": ["claim:assignment.treatment_column", "claim:assignment.treated_level"]
  },
  "story": "the person's account, verbatim, or null",
  "facts": {
    "claim:assignment.kind":           {"value": "own_choice", "status": "confirmed", "source": "user:turn:4", "said": "..."},
    "claim:assignment.rule":           {"value": "...", "status": "confirmed", "source": "user:turn:4", "said": "..."},
    "claim:assignment.depends_on":     {"value": ["lunch", "parental_level_of_education"], "status": "confirmed", "source": "user:turn:4", "said": "..."},
    "claim:assignment.offer_column":   {"value": "none", "status": "confirmed", "source": "user:turn:6", "said": "..."},
    "claim:assignment.uptake_column":  {"value": "none", "status": "confirmed", "source": "user:turn:6", "said": "..."},
    "claim:assignment.movable":        {"value": null, "status": "empty"},
    "claim:change.what":               {"value": "...", "status": "confirmed", "source": "user:turn:2", "said": "..."},
    "claim:grain.row_is":              {"value": "one student", "status": "confirmed", "source": "user:turn:2", "said": "..."},
    "claim:grain.panel":               {"value": false, "status": "confirmed", "source": "data"},
    "claim:sampling.how":              {"value": "all", "status": "confirmed", "source": "user:turn:7", "said": "..."},
    "claim:missing.why":               {"value": "none", "status": "confirmed", "source": "data"},
    "claim:unobserved.exists":         {"value": false, "status": "confirmed", "source": "user:turn:9", "said": "..."},
    "claim:exclusion.exists":          {"value": null, "status": "empty"},
    "claim:mediator.exists":           {"value": null, "status": "unknown", "source": "user:turn:11", "said": "don't know"},
    "claim:spillover.possible":        {"value": false, "status": "confirmed", "source": "user:turn:10", "said": "..."},
    "col:lunch.meaning":               {"value": "...", "status": "confirmed", "source": "user:turn:3", "said": "..."},
    "col:lunch.when":                  {"value": "before", "status": "confirmed", "source": "user:turn:3", "said": "..."},
    "col:lunch.feeds_treatment":       {"value": true, "status": "confirmed", "source": "code:depends_on"},
    "col:lunch.moves_outcome":         {"value": null, "status": "empty"},
    "col:lunch.moved_by_change":       {"value": false, "status": "confirmed", "source": "user:turn:3", "said": "..."},
    "col:lunch.measures_outcome":      {"value": null, "status": "empty"},
    "col:lunch.same_as":               {"value": "none", "status": "drafted", "source": "model:roles", "reason": "the roles rung read it so"},
    "col:lunch.nested_in":             {"value": "none", "status": "drafted", "source": "model:roles", "reason": "the roles rung read it so"},
    "col:lunch.stands_for":            {"value": "household income", "status": "drafted", "source": "model:roles", "reason": "..."},
    "col:lunch.may_modify":            {"value": true, "status": "confirmed", "source": "user:turn:12", "said": "..."},
    "col:reading_score.when":          {"value": "after", "status": "confirmed", "source": "user:turn:5", "said": "..."},
    "col:reading_score.measures_outcome": {"value": true, "status": "confirmed", "source": "user:turn:5", "said": "..."}
  },
  "columns": [
    {"name": "lunch", "key": "lunch", "role": "depends_on",
     "profile": {"kind": "categorical", "distinct": 2, "nulls": 0, "null_rate": 0.0, "constant": false,
                 "top_values": [{"value": "standard", "count": 645, "share": 0.645}, {"value": "free/reduced", "count": 355, "share": 0.355}]}},
    {"name": "math score", "key": "math_score", "role": "outcome", "profile": {"kind": "numeric", "distinct": 81, "nulls": 0, "numeric": {"min": 0, "p50": 66, "max": 100, "mean": 66.1}}}
  ],
  "probes": [
    {"address": "probe:adjustment.arms", "value": 358, "passed": true, "detail": "358 completed, 642 none"},
    {"address": "probe:adjustment.overlap", "value": 1.0, "passed": true, "detail": "both arms in every cell of lunch x parental level of education"},
    {"address": "probe:data.by_arm.lunch", "value": null, "passed": null, "detail": "free/reduced: 41% of completed, 33% of none"}
  ],
  "said": [
    {"turn": 3, "about": "col:lunch.when", "text": "the district sets it from household income before the exam"},
    {"turn": 14, "about": "lane:claim:mediator.exists", "text": "no"}
  ],
  "block": {
    "kind": "adjustment",
    "adjustment_candidates": {"value": ["lunch", "parental_level_of_education"], "rests_on": ["claim:assignment.depends_on", "col:lunch.when", "col:parental_level_of_education.when"]},
    "forbidden":             {"value": ["reading_score", "writing_score"], "rests_on": ["col:reading_score.when", "col:writing_score.when"]},
    "instrument":            {"value": null, "rests_on": ["claim:exclusion.exists"]},
    "mediator":              {"value": null, "rests_on": ["claim:mediator.exists"]},
    "unobserved_confounding": {"value": false, "rests_on": ["claim:unobserved.exists"]},
    "voluntary_uptake":      {"value": true, "rests_on": ["claim:assignment.kind"]}
  },
  "brief": {
    "road": "backdoor",
    "target": "average",
    "decisions": [{"name": "mechanism", "choice": "...", "rests_on": ["claim:assignment.rule", "claim:assignment.depends_on"], "reason": "..."}],
    "threats": [{"reason": "...", "cites": ["claim:unobserved.exists"]}],
    "bets_on": "nothing beyond lunch and parents' education drove both the course and the score"
  },
  "prior": {
    "design_id": 1,
    "run_dir": ".artifacts/runs/students-dowhy-9baefe8d",
    "seals": {"pair": {"...": "see The seal below"}, "mechanism": null}
  }
}
```

What this drops from today's `Handoff`: `assignment`, `change`, `grain`, `sampling`, `missing` as bare dicts;
`claims`; `beliefs`, `unknowns`, `contradictions` as separate lists (each is a view over `facts` by status, computed in
`case`); `treated_level` and `control_level` as loose fields (they are `pair`); `relevant_columns`, `reasons`, `why`,
`over`, `supported_now`, `chosen_assumption` (the decision's record lives in `designs/<n>/decision.json`; the lane reads
`brief.bets_on`); the claim values on `ColumnBrief` (they are `facts`). `Belief.known()` accepting `drafted` goes with
the `beliefs` dict (audit A2).

What this adds: `pair.rests_on`, `facts` with status on every entry, `rests_on` on every block field, `prior`.

One consequence for the desk: `depends_on` and `feeds_treatment` are one claim at two addresses. The desk's consistency
rules write `col:<k>.feeds_treatment = true` with source `code:depends_on` for every column `depends_on` names, so a
rung reads one address and finds the fact whichever way the person said it.

## How the reasoning follows

The order is the one ADR 0009 states. Each rung rests on the rungs below it and on declared pack addresses, and nothing
else. "Set by the pack" means every address the decider needs is `confirmed`; a draft or an empty field hands the rung
to a judgement, which sees the draft as the last reading.

1. **The pair.** Which column records the change, which level means treated, which level it is compared against, for
   whom the effect is wanted. Everything above is relative to this pair. Set by the pack when `pair.treated_level` is
   confirmed and a control level exists; else a judgement over the treatment's observed levels.
2. **The mechanism.** What set the treatment: the kind, the columns the decision or the offer looked at, whether an offer
   and a taking are two columns, whether units could move their own assignment. Rests on the pair. Set by the pack when
   the kind is a lottery or the drivers are confirmed, at either address. This rung decides which columns are parents of
   the treatment before any column is placed, whether the target is the treated (a declaration keyed on
   `self_selection`, in `checks.yaml`, not a line of code), and the compliance threat.
3. **Time.** Every other column before, at, after the change, or unknown. Rests on the pair. Code over confirmed `when`
   facts only; a drafted or empty `when` goes under unknown and is placed by the roles rung with the draft shown.
4. **The roles, together.** Every column fixed before the change, or of unknown timing, that the pack leaves open: the
   four claims against the pair, what it stands for, whether it carries the same information as another or sits inside
   one, whether the effect could differ by it. Rests on the mechanism (a driver's `affects_treatment` is settled) and
   time. A column every claim of which is confirmed never reaches the model. This rung decides the parents of the outcome
   and so the adjustment set, which of two columns carrying one thing is kept, and the modifier candidates.
5. **The post-treatment roles, together.** Every column set at or after the change that the pack leaves open: a mediator,
   another measure of the outcome, a consequence of the treatment or of the outcome, a background attribute recorded
   late, or unrelated. Rests on time and the mechanism. Decides what is excluded from the graph and whether a mediator
   path exists.
6. **The graph.** Not a rung: a pure function of the sealed rungs below, the confirmed hidden factor, and the revisions
   applied. See the next section.
7. **The road.** The engine finds every road the graph opens. The brief's road is taken when open; one open road is
   taken by code; several open roads are a judgement over the hidden factors, the mechanism and the roles. No open road
   while a hidden factor stands asks the desk the one question that could open one, when that belief is not settled;
   when the person has said there is neither, the back door is taken with the hidden factor as a sensitivity range.
   Rests on the graph and the beliefs. The road is set here and nowhere else.
8. **Heterogeneity and the target.** The candidates are the columns the roles rung or the person marked; a judgement
   names at most the declared number, each cited; the target follows from the scope and the mechanism's declaration.
9. **The threats.** Code, from the pack and the rungs below: selection into the file, the outcome's timing, an offer that
   differs from the taking, choice with the hidden factors unsettled, gaps whose reason is not settled.
10. **Checks, assessment, the pick, the freeze, the run, the attack, the answer.** As today, with two changes: a check that
    flagged a column only ever names a column (audit A8), and the freeze may not change the road (audit A6).

The contract, per rung. `reads` are patterns over the pack's addresses and the ladder's lines; the harness expands them
against the pack to compute the seal's inputs. This is the file `families/adjustment/lane/knowledge/rungs.yaml` becomes;
it is shown as JSON here because the pack is.

```json
{
  "rungs": [
    {
      "name": "pair",
      "level": "what is compared",
      "decider": {"pack": "pair.treated_level confirmed and a control level exists", "judgement": "otherwise"},
      "reads": ["pair.*", "scope.target", "col:<treatment>.profile.top_values", "claim:story.text"],
      "writes": ["ladder:pair.outcome", "ladder:pair.treatment", "ladder:pair.contrasts", "ladder:pair.target"],
      "gate": ["every level is observed", "control differs from treated", "no duplicate contrast", "every cite resolves"],
      "tools": ["describe"],
      "budget": 2
    },
    {
      "name": "mechanism",
      "level": "how assignment happened",
      "decider": {"pack": "claim:assignment.kind is lottery, or claim:assignment.depends_on is confirmed, or any col:*.feeds_treatment is confirmed true", "judgement": "otherwise"},
      "reads": ["ladder:pair.*", "claim:assignment.kind", "claim:assignment.rule", "claim:assignment.depends_on", "claim:assignment.offer_column", "claim:assignment.uptake_column", "claim:assignment.movable", "col:*.feeds_treatment", "claim:story.text", "col:*.profile.*"],
      "writes": ["ladder:mechanism.kind", "ladder:mechanism.drivers", "ladder:mechanism.offer", "ladder:mechanism.self_selection", "ladder:mechanism.reason"],
      "gate": ["kind is the pack's, never the model's", "every driver is a column in play", "offer and uptake are columns in play and differ", "no self-selection under a lottery", "every cite resolves; a refusal lists the addresses the rung may cite"],
      "tools": ["describe", "cells", "association"],
      "budget": 4
    },
    {
      "name": "timing",
      "level": "the structure",
      "decider": {"code": "always"},
      "reads": ["ladder:pair.*", "col:*.when"],
      "writes": ["ladder:timing.before", "ladder:timing.at", "ladder:timing.after", "ladder:timing.unknown"],
      "gate": ["a confirmed when is placed; a drafted or empty when is placed under unknown"]
    },
    {
      "name": "roles",
      "level": "what else touches the outcome",
      "decider": {"pack": "per column: every one of the four claims confirmed, or the column is the named instrument or mediator, or the person called it a measure of the outcome", "judgement": "every other column under before or unknown, all together"},
      "reads": ["ladder:timing.before", "ladder:timing.unknown", "ladder:mechanism.drivers", "block.instrument", "block.mediator", "col:<k>.when", "col:<k>.feeds_treatment", "col:<k>.moves_outcome", "col:<k>.moved_by_change", "col:<k>.measures_outcome", "col:<k>.same_as", "col:<k>.nested_in", "col:<k>.stands_for", "col:<k>.may_modify", "col:<k>.meaning", "col:<k>.profile.*", "probe:data.redundancy.*", "claim:story.text", "said:*"],
      "writes": ["ladder:roles.<k>"],
      "gate": ["every listed column once and nothing else", "a settled claim is copied with its address", "a claim marked true rests on a cited reason", "not both feeding and fed by the treatment", "no contradiction with a confirmed fact unless the pack marks it contested and it is cited", "a departure from the last reading names a Departure with a cite", "redundant_with and nested_in name another column in play and rest on a redundancy fact or a same_as or nested_in line", "every cite resolves; a refusal lists the addresses the rung may cite"],
      "tools": ["describe", "by_arm", "association", "redundancy", "cells", "timing"],
      "budget": 8
    },
    {
      "name": "post_roles",
      "level": "what else touches the outcome",
      "decider": {"pack": "as roles", "judgement": "every other column under at or after, all together"},
      "reads": ["ladder:timing.at", "ladder:timing.after", "ladder:mechanism.*", "block.mediator", "col:<k>.when", "col:<k>.moved_by_change", "col:<k>.measures_outcome", "col:<k>.moves_outcome", "col:<k>.meaning", "col:<k>.profile.*", "claim:story.text", "said:*"],
      "writes": ["ladder:post_roles.<k>"],
      "gate": ["every listed column once", "a cite per column", "a settled claim is copied", "no contradiction with a confirmed fact", "no unexplained departure", "the outcome by arm is refused"],
      "tools": ["describe", "association", "timing"],
      "budget": 4
    },
    {
      "name": "graph",
      "level": "the structure",
      "decider": {"code": "a pure function of the rungs below, claim:unobserved.exists, block.forbidden, and the revisions applied"},
      "reads": ["ladder:pair.*", "ladder:roles.*", "ladder:post_roles.*", "block.forbidden", "block.mediator", "claim:unobserved.exists", "ladder:road.hidden_factor", "revisions"],
      "writes": ["design.graph"],
      "gate": ["see The graph cannot be built wrong; the builder refuses and never repairs"]
    },
    {
      "name": "road",
      "level": "the identifying judgement",
      "decider": {"pack": "design.brief.road is open", "code": "exactly one road is open", "ask": "none is open while a hidden factor stands and claim:mediator.exists or claim:exclusion.exists is not settled", "judgement": "more than one road is open"},
      "reads": ["design.graph", "design.brief.road", "claim:unobserved.exists", "claim:exclusion.*", "claim:mediator.*", "said:* about lane:claim:mediator.exists or lane:claim:exclusion.exists", "ladder:mechanism.*", "ladder:roles.*"],
      "writes": ["ladder:road.taken", "ladder:road.open", "ladder:road.hidden_factor", "ladder:road.why", "design.estimand"],
      "gate": ["the road taken is open", "a reason is given", "every cite resolves", "nothing above this rung changes the road"],
      "tools": [],
      "budget": 0
    },
    {
      "name": "heterogeneity",
      "level": "where the effect differs",
      "decider": {"code": "no candidate", "judgement": "candidates exist"},
      "reads": ["ladder:roles.* (modifier_candidate)", "col:*.may_modify", "ladder:timing.before", "ladder:mechanism.self_selection", "scope.target", "checks.yaml: modifiers, target_units"],
      "writes": ["ladder:heterogeneity.modifiers", "ladder:heterogeneity.target", "ladder:heterogeneity.why"],
      "gate": ["every modifier is a candidate, once", "at most the declared number", "a cite per modifier", "every cite resolves", "the outcome by arm is refused"],
      "tools": ["describe", "by_arm"],
      "budget": 4
    },
    {
      "name": "threats",
      "level": "the risks",
      "decider": {"code": "always"},
      "reads": ["claim:sampling.how", "col:<outcome>.when", "col:*.profile.nulls", "claim:missing.why", "ladder:mechanism.offer", "ladder:mechanism.self_selection", "claim:unobserved.exists"],
      "writes": ["ladder:threats.<name>"]
    },
    {
      "name": "checks",
      "level": "how to compute",
      "decider": {"code": "always"},
      "reads": ["design.graph", "design.estimand", "block.adjustment_candidates", "ladder:threats.*", "ladder:*.unsure", "facts by status: unknown, refuted, contradiction", "beliefs.yaml", "checks.yaml"],
      "writes": ["check:<contrast>.<name>", "decline:<stage>.<about>"]
    },
    {
      "name": "assess",
      "level": "how to compute",
      "decider": {"code": "beliefs.yaml decides a stop, an ask, or a level edit", "judgement": "a flag stands after that"},
      "reads": ["design.graph", "design.estimand", "check:*", "ladder:*"],
      "writes": ["assessment"],
      "gate": ["proceed is refused under a hard flag", "a revision names a column a flag names", "every cite is a check, pack or ladder address", "three revisions at most"]
    },
    {
      "name": "pick",
      "level": "how to compute",
      "decider": {"code": "estimators.yaml filtered by the road taken, the treatment kind, the outcome kind, the adjustment set", "judgement": "one of the survivors"},
      "reads": ["ladder:road.taken", "design.estimand", "check:*", "estimators.yaml"],
      "writes": ["design.estimator", "design.also_run"],
      "gate": ["the name is offered", "every cite is a check, pack, ladder or estimator address"]
    },
    {
      "name": "freeze",
      "level": "freeze, run, attack, answer",
      "decider": {"code": "always"},
      "reads": ["ladder:*", "design.graph", "design.estimand", "design.estimator", "check:*", "refuters.yaml"],
      "writes": ["design.json"],
      "gate": ["the road is the road rung's", "every refuter whose conditions match is listed", "analyse reads design.json and nothing else"]
    }
  ]
}
```

## The graph cannot be built wrong

The question was how to keep a graph from being built in an invalid configuration. The answer is that the builder
never decides anything. Every edge is a transcription of one sealed line with its cite, and every refusal happens at
the rung that creates the line, before any graph exists. The builder refuses what it cannot transcribe; it never repairs.

| invariant | held where |
|---|---|
| every node is a column the intake loaded, or the one hidden node | the builder refuses |
| the graph is acyclic | the builder refuses; `verify_graph` folds into it |
| every edge carries the cites of the relation it came from; the hidden node's edges cite `claim:unobserved.exists` | the builder refuses an edge without a cite |
| a column cannot both feed the treatment and be changed by it | the roles and post-roles gates |
| a column the block forbids is never a node; a block field rests on confirmed facts only, so a drafted `after` is a post-role to judge, not an exclusion | the block, and the builder |
| a measure of the outcome is never a node | the roles and post-roles records; the builder excludes with the reason |
| the mediator path replaces the direct edge only when `claim:mediator.exists` is confirmed true and names a column in play | the builder reads the confirmed fact, never a draft |
| the hidden node exists if and only if `claim:unobserved.exists` is confirmed true; a draft is a flag, not a node | the builder |
| every column in play has exactly one relation | the builder stops on a column with none |
| the road is set once, at the road rung; the pick is offered that road's estimators; the freeze copies it | the freeze's road switch is deleted |
| the builder is pure: the same seals and revisions give the same graph; a revision re-enters at the builder, never edits an edge in place | a test |
| a rung writes its own line and nothing below it; when the road rung leaves the hidden factor in as a range, it writes `ladder:road.hidden_factor` and the builder rebuilds, rather than the road rung rewriting the graph | a test over every node's update |

## The seal and the replay

A seal is written the moment a rung's gate accepts its record, to `<run_dir>/ladder.json`, the whole file each time,
written to a temporary name and renamed. It holds, per rung:

```json
{
  "rung": "roles",
  "by": "judgement",
  "design_id": 1,
  "memory_version": 62,
  "inputs": {
    "ladder:timing.before": "lunch, parental_level_of_education, gender, race_ethnicity",
    "ladder:mechanism.drivers": "lunch, parental_level_of_education",
    "col:lunch.when": "before·confirmed",
    "col:lunch.feeds_treatment": "true·confirmed",
    "col:gender.moves_outcome": "·empty",
    "probe:data.redundancy.gender.race_ethnicity": "0.02"
  },
  "record": {"items": [{"column": "gender", "affects_treatment": false, "affects_outcome": true, "...": "..."}], "unsure": []},
  "episode": {"node": "roles", "facts": [{"n": 1, "tool": "redundancy", "args": {"a": "gender", "b": "race_ethnicity"}, "text": "..."}], "refusals": [], "calls": 1, "tries": [{"n": 1, "errors": []}]},
  "sealed_at": "2026-10-05T14:02:11Z"
}
```

Inputs are the value and the status, written out, not hashed: a person can read why a rung reran. A rung computed by
code seals too, with `by = "code"` or `"pack"`, so a rung above can replay against it.

The replay rule, applied in ladder order:

1. A rung replays when its seal exists in `prior`, every rung below it replayed or computed to a record equal to its own
   seal, every input in the seal has the same value and status in the new pack, and every cite in the sealed record
   still resolves in the new pack, the ladder so far, or the sealed episode.
2. A replayed rung keeps its record and its episode log, so `probe:<rung>.<n>` cites stand, and is written with
   `by = "replay"` and `from = "design:<prior.design_id>"`. The report says which rungs were replayed.
3. The first rung that does not replay runs, and every rung above it runs. Nothing after the freeze is replayed.
4. A stop seals nothing for the failing rung. The next run on the same pack replays every rung below and retries the
   one that failed. A gate refusal for a missing or unresolved cite lists the addresses the rung may cite, so a retry is
   not blind.
5. The desk builds `prior` from the last design of the same dataset with the same pair, reading `ladder.json` under that
   design's `run_dir`. A new pair has no prior, as `forget_change` already drops the fields. A what-if fork carries the
   base design's seals, so only the rungs the supposition touches run.
6. The ask-back loop closes: the road rung asks, the lane stops with the five rungs below sealed, the person answers, the
   next design replays those five and runs the road.

The run's reasoning is thereby stable across runs for as long as its inputs stand, which is what a checkpoint is for.
`absorb_relations` stays: it is a different thing, the person confirming a run's reading so the next run takes it as a
fact, which then shows in the seal's inputs as a status change.

## What goes

Code the audit or this policy finds dead or doubled, deleted rather than kept:

- `Estimand.alternatives` and `Estimand.dowhy_text`
  ([contracts.py:263](../../causal_agent/families/adjustment/lane/contracts.py)): written by `identify`, read nowhere.
- `Design.params` as a copy of the catalogue entry while `analyse` read the entry again: the frozen design now carries
  the catalogue entries it runs (`estimator`, `also_run`, `refuters`) and the rules the within-level runs follow, and nothing
  after the freeze reads the yaml. Built 2026-10-05.
- The road switch in `freeze_design` ([nodes.py:1272](../../causal_agent/families/adjustment/lane/nodes.py)):
  unreachable once the road rung sets the road and the pick is offered that road alone; audit A6.
- `Handoff.claims`, `reasons`, `why`, `over`, `supported_now`, `relevant_columns`, `unknowns`, `contradictions`,
  `beliefs`, and the bare `assignment`, `change`, `grain`, `sampling`, `missing` dicts: replaced by `facts`, `pair` and
  the decision record on disk.
- `ColumnBrief.set_by` and `proxy`, and `derived_from` in `fields.yaml`: no rung reads them; `set_by` is in `case.FIELDS`
  only to be skipped. A decision rests on them or they leave the catalogue.
- `SpecialistState.interpret_attempts` ([state.py:41](../../causal_agent/families/adjustment/lane/state.py)): set once,
  never read. `InterpretTask.errors`: always empty, never read.
- The three readings of a column's timing (audit C2) collapse to one: confirmed `when`, else unknown.
- `fact_relation`'s default of `affects_outcome = True` for a driver fixed before the change (audit B2): the claim is
  open and the roles rung judges it, with `affects_treatment` settled.

## The tests that hold it

- **Replay makes no call.** A pack whose `prior` is its own run's seals climbs to the freeze with a fake that raises on
  any model call.
- **A rung reads only what it declares.** For each rung, every address outside its `reads` is scrambled and the record is
  unchanged.
- **A moved input reruns the right rungs.** Change one input of the roles rung: the pair, the mechanism and time replay;
  the roles and everything above run.
- **The builder is pure and refuses.** The same seals give the same graph; a cyclic or uncited edge, a forbidden node, a
  column with no relation, each stops the builder.
- **No field is dead.** Every pack field and every `rungs.yaml` key is read by a lane or the desk, in the style of
  `test_core_names_no_family`.
- **A refusal names addresses.** A cite gate error lists the resolvable addresses for that rung.

## What it costs

One more file per run and one more field in the pack. A `rungs.yaml` per lane that a test must keep honest, the price of
knowing what a rung reads. The desk must find the previous design of the same pair at hand-off, which `designs/<n>/` already
holds. The pack changes shape, and by the project's first rule the old shape is deleted, not read: the stored hand-offs under
`evals/handoffs/` are rewritten, and the diff-in-diff and discontinuity lanes take the same pack shape in their own ADR, with
their own `rungs.yaml`.
