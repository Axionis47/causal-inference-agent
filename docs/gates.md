# The gates

Every model call in the system, what it is given, the shape it must return, what code checks, and what happens when the check
fails. This is the page that proves the thesis: the model answers closed questions, code decides.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/judgement-relate-dark.svg">
  <img alt="One relate judgement opened up: the prompt, the structured answer, and the gate's six checks" src="diagrams/judgement-relate-light.svg">
</picture>

The picture is one relate call from the checked-in run under `docs/demo/run/`: the prompt rendered again from the stored pack
by the lane's own helpers, the answer as the frozen graph records it, and the gate's checks run again on that answer.

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
| `mine`, `infer` (the Reader) | the note or the message, the open fields with their frames | `Reading`: field updates each with the words it rests on, confirms, unknowns, focus, a question, a draw request | `ops.apply`: a source on every update, a model may only draft, a belief only from the person, values fit the field; a refused update is re-prompted | after three tries the surviving updates land and the rest are dropped |
| `prefilter` (wide tables only) | the question, the change, one column card | `PrefilterVote` | a vote per column, code keeps the relevant ones | a column with no vote stays in |
| `explain`, `turn` (the Explainer) | the material as addressed lines, the message | `AfterReply`: kind, text, cites, every number with its address, updates, question, draw, figure | `explainer.gate` ([desk/explainer.py:121](../causal_agent/desk/explainer.py)): the kind is legal in this phase; every cite is in the material; at least one cite; every number sits at its address within one percent, or appears in that line's text; no number in the text without an address; a figure named is a figure address; revise and what_if carry updates; requestion carries the question; draw carries the ask | three tries, then the honest fallback reply |
| `decide` (only when more than one family stands) | the families' knowledge, the fit, the probes | `FamilyDecision`: admissible, chosen, rejected with reasons, cites | `decide.gate` ([desk/nodes/decide.py:179](../causal_agent/desk/nodes/decide.py)): every cite resolves; the chosen family is admissible; admissible agrees with the fit; every registered family is accounted for; the outcome and cause are columns in the relevant set | three tries, then the design and the hand-off record the failed gate |
| `design` (the Designer) | the memory, the matrix, the family's decisions | `DesignBrief`: road, target, one `DecisionMade` per decision the family lists, threats with cites, checks, the sentence it bets on | `designer.check` ([desk/designer.py:28](../causal_agent/desk/designer.py)): the right family; every listed decision filled once; every cite resolves; a road only where the family has one | three tries, then `fallback`: every decision undecided, resting on the change card, betting on the family's stated assumption |
| `draw`, `draw_after` (the Drawer) | the ask, what is known, the columns | `DrawCode`: plan, script, caption, fact names | the script parses; the sandbox run leaves a non-empty PNG; `facts.json` is all numbers; every promised name is present | three tries, then a `Decline` and no folder |

## The adjustment lane

| node | given | returns | the gate checks | on failure |
|---|---|---|---|---|
| `contrast` (only if the pack does not name the treated level) | the question, the scope, the treatment card, the observed levels | `Contrasts`: control and treated pairs | every level is observed; control differs from treated; no duplicate pair | three tries, then a Feasibility stop |
| `relate` (one call per column the pack leaves open) | the question, the frame, the treatment and outcome cards, one column's card, what the pack settled for it | `Relation`: four yes/no, a cited reason per claim marked true | `verify_graph` ([families/adjustment/lane/nodes.py:448](../causal_agent/families/adjustment/lane/nodes.py)): a claim marked true has a reason; every reason cites; every cite resolves; not both feeding and fed by the treatment; no contradiction with a pack fact; no unexplained departure from the last run's reading; the whole graph is acyclic | the failing columns are re-sent with their errors; three rounds, then a stop |
| `assess` (only when a check flagged) | the frame, the graph, the adjustment set, the flagged checks with numbers | `DesignAssessment`: proceed, revise or stop, revisions, cites | proceed is refused while a hard flag stands; a revision must name a flagged column; every cite is a check or pack address | revise loops to `merge_graph`, three revisions at most; stop is honest |
| `pick_estimator` | the design facts, the estimators that apply with what each assumes, the ranked preferences | `EstimatorPick`: one name, why, cites | the name is in the filtered list; cites are check or pack addresses | a failed fit later excludes the pick and asks once more |
| `interpret` (one call per contrast) | the material as addressed lines, the addresses it may cite, the ones it must | `Interpretation`: the answer, the caveats, the effect stated, cites | every cite is allowed; every required address is cited, the flagged checks and the interval among them; the effect stated matches the estimate within one percent | three tries; a failed gate is written into the report |

Two rules that are code, not judgement: the checks' thresholds are declared in
[checks.yaml](../causal_agent/families/adjustment/lane/knowledge/checks.yaml), and every refuter whose conditions in
[refuters.yaml](../causal_agent/families/adjustment/lane/knowledge/refuters.yaml) match the design runs. The model never picks
the tests its own design faces.

The other two lanes have the same shape with their own judgements: diff-in-diff adds `groups` and `periods`, discontinuity adds
`score`. See [lanes.md](lanes.md).

## Why this shape

[ADR 0002](adr/0002-code-computes-facts-the-model-judges-the-harness-gates.md): code computes every fact, the model judges, the
harness gates. The cost is that no single call holds the whole story; each is asked a bounded question and checked hard. The
gain is that every answer can be refused, re-asked, and cited, and that a test can script every one of them.
