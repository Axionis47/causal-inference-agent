# The gates

Every model call in the system, what it is given, the shape it must return, what code checks, and what happens when the check
fails. This is the page that proves the thesis: the model answers closed questions, code decides.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/judgement-relate-dark.svg">
  <img alt="One column of the roles rung opened up: the prompt, the structured answer, and the gate's checks" src="diagrams/judgement-relate-light.svg">
</picture>

The picture is one column of the roles rung, from the checked-in run under `docs/demo/run/`: the prompt rendered again from the
stored pack by the lane's own helpers, the answer as the frozen graph records it, and the gate's checks run again on that answer.

## How every judgement is made

One helper, `structured` ([common/llm.py:79](../causal_agent/common/llm.py)): a system prompt, a user prompt, and a Pydantic
schema the model must fill. It returns the parsed object and the model's thought summary. The model is Gemini 2.5 Flash on
Vertex AI at temperature zero; the wrapper is the only file that names it. Tests swap it with `set_llm`.

Around every call: a loop of at most three tries, a gate written in code, and the gate's errors appended to the next prompt under
"PREVIOUS ANSWER WAS REJECTED". After three failures the node takes an honest fallback or stops with a typed record, never a guess.

## The desk

| node | given | returns | the gate checks | on failure |
|---|---|---|---|---|
| `read_question` | the question, the dataset digest, the column index | `QuestionFrame`: intent, outcome, cause, scope, relevant columns | `validate` ([desk/nodes/question.py:91](../causal_agent/desk/nodes/question.py)): the intent is an effect of a change, the outcome is a column that varies, the cause is a column that varies and is not the outcome | the question is refused with the test it failed and asked again |
| `mine`, `infer` (the Reader) | the note or the message, the open fields with their frames | `Reading`: field updates each with the words it rests on, confirms, unknowns, focus, a question, a draw request | `ops.apply`: a source on every update, a model may only draft, a belief only from the person, the story only from the desk and verbatim, values fit the field; a refused update is re-prompted | after three tries the surviving updates land and the rest are dropped |
| `prefilter` (wide tables only) | the question, the change, one column card | `PrefilterVote` | a vote per column, code keeps the relevant ones | a column with no vote stays in |
| `explain`, `turn` (the Explainer) | the material as addressed lines, the message | `AfterReply`: kind, text, cites, every number with its address, updates, question, draw, figure | `explainer.gate` ([desk/explainer.py:121](../causal_agent/desk/explainer.py)): the kind is legal in this phase; every cite is in the material; at least one cite; every number sits at its address within one percent, or appears in that line's text; no number in the text without an address; a figure named is a figure address; revise and what_if carry updates; requestion carries the question; draw carries the ask | three tries, then the honest fallback reply |
| `decide` (only when more than one family stands) | the families' knowledge, the fit, the probes | `FamilyDecision`: admissible, chosen, rejected with reasons, cites | `decide.gate` ([desk/nodes/decide.py:179](../causal_agent/desk/nodes/decide.py)): every cite resolves; the chosen family is admissible; admissible agrees with the fit; every registered family is accounted for; the outcome and cause are columns in the relevant set | three tries, then the design and the hand-off record the failed gate |
| `design` (the Designer) | the memory, the matrix, the family's decisions | `DesignBrief`: road, target, one `DecisionMade` per decision the family lists, threats with cites, checks, the sentence it bets on | `designer.check` ([desk/designer.py:28](../causal_agent/desk/designer.py)): the right family; every listed decision filled once; every cite resolves; a road only where the family has one | three tries, then `fallback`: every decision undecided, resting on the change card, betting on the family's stated assumption |
| `draw`, `draw_after` (the Drawer) | the ask, what is known, the columns | `DrawCode`: plan, script, caption, fact names | the script parses; the sandbox run leaves a non-empty PNG; `facts.json` is all numbers; every promised name is present | three tries, then a `Decline` and no folder |

## The adjustment lane

| node | given | returns | the gate checks | on failure |
|---|---|---|---|---|
| `pair` (rung 0; only if the pack does not name the treated level) | the question, the scope, the treatment card, the observed levels; may describe the treatment | `Contrasts`: control and treated pairs | every level is observed; control differs from treated; no duplicate pair | three tries inside the episode, then a Feasibility stop |
| `mechanism` (rung 1; only when the pack names no drivers) | the case, the treatment card, the column index; may describe, tabulate cells, or ask a candidate's association with the treatment | `Mechanism`: drivers, an offer and an uptake column or null, whether units could move themselves, cites | every driver is a column in play; an offer or uptake named is a column; the two differ; no self-selection under a lottery; cites resolve in the pack, the facts asked for, or the rungs below | three tries, then a stop |
| `roles` (rung 3; one episode over every pre-treatment column the pack leaves open) | the case with the ladder so far, the pair's cards, one block per column with what the pack settled and the last reading; may look by arm, at associations, redundancy, cells | `Roles`: one `Role` per column, the four claims with a cited reason each, what it stands for, `redundant_with`, `nested_in`, a modifier candidate, links, departures | `roles` gate ([families/adjustment/lane/nodes.py](../causal_agent/families/adjustment/lane/nodes.py)): every listed column once and nothing else; the settled claims copied; a claim marked true has a cited reason; every cite resolves; not both feeding and fed by the treatment; no contradiction with a pack fact; a departure from the last reading names a `Departure` with a cite; a redundancy or nesting names another column in play and rests on a redundancy fact; the outcome by arm is refused before the answer is asked | the errors go back with the log kept, three tries, then a stop |
| `post_roles` (rung 4; one episode over every at-or-after column the pack leaves open) | as `roles`; may describe or ask a column's association with the treatment, never with the outcome | `PostRoles`: one `PostRole` per column with its kind and a cited reason | every listed column once; a cite per column; cites resolve; the settled claims copied; no contradiction; no unexplained departure | three tries, then a stop |
| `verify_graph` (code, no model) | the merged graph | | acyclic; every node a table column; a role for every column | an honest stop |
| `assess` (only when a check flagged) | the frame, the graph, the adjustment set, the flagged checks with numbers | `DesignAssessment`: proceed, revise or stop, revisions, cites | proceed is refused while a hard flag stands; a revision must name a flagged column; every cite is a check or pack address | revise loops to `merge_graph`, three revisions at most; stop is honest |
| `pick_estimator` | the design facts, the estimators that apply with what each assumes, the ranked preferences | `EstimatorPick`: one name, why, cites | the name is in the filtered list; cites are check or pack addresses | a failed fit later excludes the pick and asks once more |
| `interpret` (one call per contrast) | the material as addressed lines, the addresses it may cite, the ones it must | `Interpretation`: the answer, the caveats, the effect stated, cites | every cite is allowed; every required address is cited, the flagged checks and the interval among them; the effect stated matches the estimate within one percent | three tries; a failed gate is written into the report |

Two rules that are code, not judgement: the checks' thresholds are declared in
[checks.yaml](../causal_agent/families/adjustment/lane/knowledge/checks.yaml), and every refuter whose conditions in
[refuters.yaml](../causal_agent/families/adjustment/lane/knowledge/refuters.yaml) match the design runs. The model never picks
the tests its own design faces.

The other two lanes keep the earlier shape for now, a one-shot `relate` per column with their own setup judgements: diff-in-diff
adds `groups` and `periods`, discontinuity adds `score`. Their ladders follow. See [lanes.md](lanes.md).

## What an episode is

A rung that needs a judgement runs as a bounded episode ([lane/episode.py](../causal_agent/lane/episode.py)). The model first
reads its material with the data tools bound and may call them, at most the budget `checks.yaml` declares for that rung (two for
the pair, four for the mechanism, eight for the roles, four for the post-treatment roles). Each result comes back as a tool message
with an address, `probe:<rung>.<n>`, and is logged. Then the facts it gathered are rendered under FACTS YOU ASKED FOR and the record
is asked for with the same structured call as every other judgement. The gate's errors go back with the log kept, and the model may
look again while budget remains; three refusals return no record and the rung stops. The tools are describe, by arm, association,
redundancy, cells and timing ([lane/tools.py](../causal_agent/lane/tools.py)); a call that would join the outcome with the treatment
is refused by code with a reason the model reads, and the refusal is logged too. The report lists every episode's calls, facts and
refusals.

## Why this shape

[ADR 0002](adr/0002-code-computes-facts-the-model-judges-the-harness-gates.md): code computes every fact, the model judges, the
harness gates. The cost is that no single call holds the whole story; each is asked a bounded question and checked hard. The
gain is that every answer can be refused, re-asked, and cited, and that a test can script every one of them.
