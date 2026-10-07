# The diff-in-diff lane's ladder

Written by `make ladder` from `causal_agent/families/diff_in_diff/lane/knowledge/` (rungs.yaml, space.yaml, checks.yaml, beliefs.yaml
and the catalogue). Do not edit; change the declarations.

The ladder climbs to `freeze_design`; every rung may go only where this table says, and a stop goes to `feasibility`.

Loops still in the graph:

- assess → merge_controls, at most 3: a control delta on a flagged check
- estimate → pick_estimator, at most 1: a second pick when the estimator failed to fit

## The rungs

| rung | level | decided by | fixes | goes on to | budget | may ask |
|---|---|---|---|---|---|---|
| load | the file | code: always |  | case · stop |  |  |
| case | the pack weighed | code: always |  | groups |  |  |
| groups | who got the change | pack: the treated group is confirmed; judgement: otherwise |  | periods · stop | 2 |  |
| periods | when the change came | pack: the period of the change is confirmed; ask: the first period at or after the change is not known; judgement: otherwise |  | shape_table · stop | 2 | claim:change.period_value |
| shape_table | the panel | code: always |  | mechanism · stop |  |  |
| mechanism | how assignment happened | judgement: always |  | trends · stop | 3 |  |
| trends | the evidence | code: always |  | comparison |  |  |
| comparison | the comparison | judgement: always |  | controls · stop | 6 |  |
| controls | what else moves the outcome | pack: a column whose claims are confirmed; judgement: every other candidate |  | merge_controls · stop | 8 |  |
| merge_controls | the design | code: a pure function of the rungs below and the revisions applied |  | verify |  |  |
| verify | the design | code: always |  | heterogeneity · stop |  |  |
| heterogeneity | where the effect differs | code: no candidate; judgement: candidates exist |  | threats · stop | 4 |  |
| threats | the risks | code: always |  | check_design |  |  |
| check_design | the checks | code: always |  | assess, pick_estimator |  |  |
| assess | the assessment | code: beliefs.yaml decides a stop or a question before any judgement; ask: a belief rule asks; judgement: a flag stands |  | pick_estimator, merge_controls · stop | 0 |  |
| pick_estimator | the pick | judgement: always |  | freeze_design · stop | 0 |  |
| freeze_design | the design | code: always |  | estimate |  |  |
| estimate | the estimate | code: always |  | placebo, interpret, pick_estimator · stop |  |  |
| placebo (worker) | the estimate | code: always |  | interpret |  |  |
| interpret | the answer | judgement: always |  | figures | 0 |  |

## The end states

Derived from the declarations: 0 answers, 4 questions, 20 stops.

### Answers

This lane declares no space yet, so its answers are not enumerated.

### Questions back

| rung | address | question |
|---|---|---|
| periods | claim:change.period_value |  |
| assess | claim:trend_continues.believed | Apart from the change, would the group that got it have kept moving with the others? |
| assess | claim:trend_continues.believed | The groups were already moving apart before the change (joint test on the periods before, p = {value}). You said: "{said}". Is there a reason they would have moved differently apart from the change? |
| assess | claim:change.period_value | Which value of the period column is the first period at or after the change? |

### Typed stops

| rung | cause | reason | what would fix it |
|---|---|---|---|
| load | intake | the hand-off names no file, or a column it names is not in the file | a hand-off whose file and columns exist |
| load | shape | the table fails a shape rule the lane declares | a table the rule holds on |
| groups | gate | the record was refused on every try | clearer notes on what the rung reads |
| periods | gate | the record was refused on every try | clearer notes on what the rung reads |
| shape_table | shape | the table fails a shape rule the lane declares | a table the rule holds on |
| mechanism | gate | the record was refused on every try | clearer notes on what the rung reads |
| comparison | gate | the record was refused on every try | clearer notes on what the rung reads |
| controls | gate | the record was refused on every try | clearer notes on what the rung reads |
| verify | graph | what the rungs built fails verification | clearer column notes about what fed the decision |
| heterogeneity | gate | the record was refused on every try | clearer notes on what the rung reads |
| assess | check:units.min_per_group | how many units got the change and how many are compared against them; min per group past 1 | a design that passes the check |
| assess | check:pre_trends.p_value | whether the two groups were already moving apart before the change; p value past 0.01 | a design that passes the check |
| assess | check:adoption.hard_when_no_estimator_applies | adoption; hard when no estimator applies | a design that passes the check |
| assess | belief:trend_continues+check:pre_trends | the test and the person agree: the groups were already moving apart before the change | a different answer at [claim:trend_continues.believed] |
| assess | threat:selection | rows were drawn by how the outcome turned out; a comparison among the selected does not reach the population the question asks about | a file drawn and measured so the threat does not stand |
| assess | threat:outcome_timing | the outcome was measured before the change; it cannot carry its effect | a file drawn and measured so the threat does not stand |
| assess | assessment | a hard flag the assessment could not answer | a design that answers it, or the person's word on it |
| assess | gate | the record was refused on every try | clearer notes on what the rung reads |
| pick_estimator | gate | the record was refused on every try | clearer notes on what the rung reads |
| estimate | engine | the estimator failed to fit | a different estimator entry |
