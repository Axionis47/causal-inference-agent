# The adjustment lane's ladder

Written by `make ladder` from `causal_agent/families/adjustment/lane/knowledge/` (rungs.yaml, space.yaml, checks.yaml, beliefs.yaml
and the catalogue). Do not edit; change the declarations.

The ladder climbs to `freeze_design`; every rung may go only where this table says, and a stop goes to `feasibility`.

Loops still in the graph:

- assess → merge_graph, at most 3: a revision delta on a flagged check
- after_analyse → pick_estimator, at most 1: a second pick when the estimator failed to fit

## The rungs

| rung | level | decided by | fixes | goes on to | budget | may ask |
|---|---|---|---|---|---|---|
| load | the file | code: always | outcome | case · stop |  |  |
| case | the pack weighed | code: always |  | pair |  |  |
| pair | what is compared | pack: the treated level is confirmed and seen in the table; judgement: otherwise | treatment, target, contrasts | mechanism · stop | 2 |  |
| mechanism | how assignment happened | judgement: always |  | timing · stop | 4 |  |
| timing | the structure | code: always |  | roles |  |  |
| roles | what else touches the outcome | pack: a column whose four claims are confirmed; judgement: every other column set before or at an unknown time |  | post_roles · stop | 8 |  |
| post_roles | what else touches the outcome | pack: as roles; judgement: every other column set at or after the change |  | merge_graph · stop | 4 |  |
| merge_graph | the structure | code: a pure function of the rungs below and the revisions applied |  | verify_graph |  |  |
| verify_graph | the structure | code: always |  | road · stop |  |  |
| road | the identifying judgement | pack: the brief's road is open on the graph and runnable; code: exactly one road is open and runnable; ask: no road is open while a hidden factor stands and the mediator or the exclusion belief is unasked; judgement: several roads are open and runnable | estimand, adjustment_set, sensitivity_required, confounders | heterogeneity · stop | 0 | claim:mediator.exists, claim:exclusion.exists, claim:mediator.column, claim:exclusion.column |
| heterogeneity | where the effect differs | code: no candidate; judgement: candidates exist | modifiers | threats · stop | 4 |  |
| threats | the risks | code: always |  | check_design |  |  |
| check_design | the checks | code: always |  | assess, pick_estimator |  |  |
| assess | the assessment | code: beliefs.yaml decides a stop or a question before any judgement; judgement: a flag stands |  | pick_estimator, merge_graph · stop | 0 |  |
| pick_estimator | the pick | judgement: always |  | freeze_design · stop | 0 |  |
| freeze_design | the design | code: always |  | analyse |  |  |
| analyse (worker) | the estimate | code: always |  | after_analyse |  |  |
| after_analyse | the estimate | code: always |  | interpret, pick_estimator, figures · stop |  |  |
| interpret (worker) | the answer | judgement: always |  | figures | 0 |  |

## The space

| dimension | values | decided by | read from | not run |
|---|---|---|---|---|
| treatment | binary | pair | probe:data.levels |  |
| outcome | continuous, binary | load | col:<outcome>.profile.kind | other: the outcome is neither a number nor a yes/no; no estimator fits it |
| estimand | backdoor, iv, frontdoor | road |  |  |
| adjustment_set | empty, nonempty | road |  |  |
| sensitivity_required | False, True | road | claim:unobserved.exists |  |
| target | ate, att, atc (soft) | pair | scope.target |  |
| contrasts | 1 to 11 | pair |  | above: more than twelve treatment levels; this lane compares levels, one contrast per level, and a numeric treatment with that many values is a dose that needs a dose-response lane |
| confounders | 0 to smallest_arm/10 | road |  | above: more adjustment columns than the smaller arm can support, at ten rows each |
| modifiers | 0 to 3 | heterogeneity |  |  |
| estimand=frontdoor+outcome=binary |  |  |  | the front-door two-stage regression fits a continuous outcome; no entry runs it on a yes/no |

## The end states

Derived from the declarations: 20 answers, 4 questions, 28 stops.

### Answers: the cells a design can reach

| cell | estimators | refuters |
|---|---|---|
| treatment=binary, outcome=continuous, estimand=backdoor, adjustment_set=empty, sensitivity_required=False | linear_regression | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=backdoor, adjustment_set=empty, sensitivity_required=True | linear_regression | placebo_treatment_refuter, data_subset_refuter, add_unobserved_common_cause |
| treatment=binary, outcome=continuous, estimand=backdoor, adjustment_set=nonempty, sensitivity_required=False | propensity_score_stratification, propensity_score_weighting, linear_regression, propensity_score_matching | placebo_treatment_refuter, random_common_cause, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=backdoor, adjustment_set=nonempty, sensitivity_required=True | propensity_score_stratification, propensity_score_weighting, linear_regression, propensity_score_matching | placebo_treatment_refuter, random_common_cause, data_subset_refuter, add_unobserved_common_cause |
| treatment=binary, outcome=continuous, estimand=iv, adjustment_set=empty, sensitivity_required=False | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=iv, adjustment_set=empty, sensitivity_required=True | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=iv, adjustment_set=nonempty, sensitivity_required=False | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=iv, adjustment_set=nonempty, sensitivity_required=True | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=frontdoor, adjustment_set=empty, sensitivity_required=False | frontdoor_two_stage | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=frontdoor, adjustment_set=empty, sensitivity_required=True | frontdoor_two_stage | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=frontdoor, adjustment_set=nonempty, sensitivity_required=False | frontdoor_two_stage | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=continuous, estimand=frontdoor, adjustment_set=nonempty, sensitivity_required=True | frontdoor_two_stage | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=binary, estimand=backdoor, adjustment_set=empty, sensitivity_required=False | generalized_linear_model | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=binary, estimand=backdoor, adjustment_set=empty, sensitivity_required=True | generalized_linear_model | placebo_treatment_refuter, data_subset_refuter, add_unobserved_common_cause |
| treatment=binary, outcome=binary, estimand=backdoor, adjustment_set=nonempty, sensitivity_required=False | propensity_score_stratification, propensity_score_weighting, generalized_linear_model, propensity_score_matching | placebo_treatment_refuter, random_common_cause, data_subset_refuter |
| treatment=binary, outcome=binary, estimand=backdoor, adjustment_set=nonempty, sensitivity_required=True | propensity_score_stratification, propensity_score_weighting, generalized_linear_model, propensity_score_matching | placebo_treatment_refuter, random_common_cause, data_subset_refuter, add_unobserved_common_cause |
| treatment=binary, outcome=binary, estimand=iv, adjustment_set=empty, sensitivity_required=False | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=binary, estimand=iv, adjustment_set=empty, sensitivity_required=True | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=binary, estimand=iv, adjustment_set=nonempty, sensitivity_required=False | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |
| treatment=binary, outcome=binary, estimand=iv, adjustment_set=nonempty, sensitivity_required=True | instrumental_variable | placebo_treatment_refuter, data_subset_refuter |

### Questions back

| rung | address | question |
|---|---|---|
| road | claim:mediator.exists |  |
| road | claim:exclusion.exists |  |
| road | claim:mediator.column |  |
| road | claim:exclusion.column |  |

### Typed stops

| rung | cause | reason | what would fix it |
|---|---|---|---|
| load | space:outcome | no value of outcome the lane runs is left | a file inside the space |
| load | space:outcome=other | the outcome is neither a number nor a yes/no; no estimator fits it | a file inside the space |
| load | intake | the hand-off names no file, or a column it names is not in the file | a hand-off whose file and columns exist |
| pair | space:treatment | no value of treatment the lane runs is left | a design inside the space |
| pair | space:contrasts | no value of contrasts the lane runs is left | a design inside the space |
| pair | space:contrasts=above | more than twelve treatment levels; this lane compares levels, one contrast per level, and a numeric treatment with that many values is a dose that needs a dose-response lane | a design inside the space |
| pair | gate | the record was refused on every try | clearer notes on what the rung reads |
| mechanism | gate | the record was refused on every try | clearer notes on what the rung reads |
| roles | gate | the record was refused on every try | clearer notes on what the rung reads |
| post_roles | gate | the record was refused on every try | clearer notes on what the rung reads |
| verify_graph | graph | what the rungs built fails verification | clearer column notes about what fed the decision |
| road | space:sensitivity_required | no value of sensitivity_required the lane runs is left | a design inside the space |
| road | space:confounders | no value of confounders the lane runs is left | a design inside the space |
| road | space:confounders=above | more adjustment columns than the smaller arm can support, at ten rows each | a design inside the space |
| road | space:estimand=frontdoor+outcome=binary | the front-door two-stage regression fits a continuous outcome; no entry runs it on a yes/no | a design inside the space |
| road | gate | the record was refused on every try | clearer notes on what the rung reads |
| heterogeneity | space:modifiers | no value of modifiers the lane runs is left | a design inside the space |
| heterogeneity | gate | the record was refused on every try | clearer notes on what the rung reads |
| assess | check:arms.min_per_arm | how many rows got the change and how many did not; min per arm past 20 | a design that passes the check |
| assess | check:overlap.common_support_share | whether both arms are found across the whole range of what the offer depended on; common support share past 0.8 | a design that passes the check |
| assess | check:overlap.max_auc | whether both arms are found across the whole range of what the offer depended on; max auc past 0.95 | a design that passes the check |
| assess | check:balance.smd | how alike the two arms are on a column before the adjustment, and after weighting; smd past 0.25 | a design that passes the check |
| assess | threat:selection | rows were drawn by how the outcome turned out; a comparison among the selected does not reach the population the question asks about | a file drawn and measured so the threat does not stand |
| assess | threat:outcome_timing | the outcome was measured before the change; it cannot carry its effect | a file drawn and measured so the threat does not stand |
| assess | assessment | a hard flag the assessment could not answer | a design that answers it, or the person's word on it |
| assess | gate | the record was refused on every try | clearer notes on what the rung reads |
| pick_estimator | gate | the record was refused on every try | clearer notes on what the rung reads |
| after_analyse | engine | the estimator failed to fit | a different estimator entry |
