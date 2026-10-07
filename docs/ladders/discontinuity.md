# The discontinuity lane's ladder

Written by `make ladder` from `causal_agent/families/discontinuity/lane/knowledge/` (rungs.yaml, space.yaml, checks.yaml, beliefs.yaml
and the catalogue). Do not edit; change the declarations.

The ladder climbs to `freeze_design`; every rung may go only where this table says, and a stop goes to `feasibility`.

Loops still in the graph:

- estimate → pick_estimator, at most 1: a second pick when the estimator failed to fit

## The rungs

| rung | level | decided by | fixes | goes on to | budget | may ask |
|---|---|---|---|---|---|---|
| load | the file | code: always |  | case · stop |  |  |
| case | the pack weighed | code: always |  | score |  |  |
| score | the score and the line | pack: the score column and the cutoff are confirmed; ask: the pack names the rule and not its column or its cutoff; judgement: otherwise |  | shape_table · stop | 2 | claim:assignment.score_column, claim:assignment.cutoff |
| shape_table | the table at the line | code: always |  | density · stop |  |  |
| density | the evidence | code: always |  | line |  |  |
| line | what the line decides | judgement: always |  | balance · stop | 4 |  |
| balance | the evidence | code: always |  | covariates |  |  |
| covariates | what else differs at the line | pack: a column whose claims are confirmed; judgement: every other candidate |  | merge_covariates · stop | 8 |  |
| merge_covariates | the design | code: a pure function of the rungs below |  | verify |  |  |
| verify | the design | code: always |  | heterogeneity · stop |  |  |
| heterogeneity | where the effect differs | code: no candidate; judgement: candidates exist |  | threats · stop | 4 |  |
| threats | the risks | code: always |  | check_design |  |  |
| check_design | the checks | code: always |  | assess, pick_estimator |  |  |
| assess | the assessment | code: beliefs.yaml decides a stop or a question before any judgement; ask: a belief rule asks; judgement: a flag stands |  | pick_estimator · stop | 0 |  |
| pick_estimator | the pick | judgement: always |  | window · stop | 0 |  |
| window | the width | code: one selector applies; judgement: several apply |  | freeze_design · stop | 2 |  |
| freeze_design | the design | code: always |  | estimate |  |  |
| estimate | the estimate | code: always |  | placebo, interpret, pick_estimator · stop |  |  |
| placebo (worker) | the estimate | code: always |  | interpret |  |  |
| interpret | the answer | judgement: always |  | figures | 0 |  |

## The end states

Derived from the declarations: 0 answers, 5 questions, 20 stops.

### Answers

This lane declares no space yet, so its answers are not enumerated.

### Questions back

| rung | address | question |
|---|---|---|
| score | claim:assignment.score_column |  |
| score | claim:assignment.cutoff |  |
| assess | claim:cutoff_only.believed | Does anything else change for a unit at the cutoff besides the change asked about? |
| assess | claim:cutoff_only.believed | Does anything else change for a unit at the cutoff besides the change asked about? |
| assess | assignment.movable | The number of units jumps at the cutoff (density test p = {value}). Could a unit have changed its score once it knew the rule? |

### Typed stops

| rung | cause | reason | what would fix it |
|---|---|---|---|
| load | intake | the hand-off names no file, or a column it names is not in the file | a hand-off whose file and columns exist |
| load | shape | the table fails a shape rule the lane declares | a table the rule holds on |
| score | gate | the record was refused on every try | clearer notes on what the rung reads |
| shape_table | shape | the table fails a shape rule the lane declares | a table the rule holds on |
| line | gate | the record was refused on every try | clearer notes on what the rung reads |
| covariates | gate | the record was refused on every try | clearer notes on what the rung reads |
| verify | graph | what the rungs built fails verification | clearer column notes about what fed the decision |
| heterogeneity | gate | the record was refused on every try | clearer notes on what the rung reads |
| assess | check:sides.min_rows | how many rows sit on each side of the cutoff; min rows past 20 | a design that passes the check |
| assess | check:density.never_hard | whether units bunch up just on one side of the line, as they would if some moved their score; never hard | a design that passes the check |
| assess | check:first_stage.no_first_stage_is_hard | how much take-up jumps at the cutoff; no first stage is hard | a design that passes the check |
| assess | belief:cutoff_only.confirmed_false | the person says something else also switches at the cutoff, so a jump there cannot be read as the change's alone | a different answer at [claim:cutoff_only.believed] |
| assess | belief:score_fixed_before.confirmed_false | the score was set at or after the decision, so the line on it cannot have decided who got the change | a different answer at [design.score_fixed_before] |
| assess | threat:selection | rows were drawn by how the outcome turned out; a comparison among the selected does not reach the population the question asks about | a file drawn and measured so the threat does not stand |
| assess | threat:outcome_timing | the outcome was measured before the change; it cannot carry its effect | a file drawn and measured so the threat does not stand |
| assess | assessment | a hard flag the assessment could not answer | a design that answers it, or the person's word on it |
| assess | gate | the record was refused on every try | clearer notes on what the rung reads |
| pick_estimator | gate | the record was refused on every try | clearer notes on what the rung reads |
| window | gate | the record was refused on every try | clearer notes on what the rung reads |
| estimate | engine | the estimator failed to fit | a different estimator entry |
