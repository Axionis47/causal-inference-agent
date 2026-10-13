# The diff-in-diff lane's ladder

Written by `make ladder` from `causal_agent/families/diff_in_diff/lane/knowledge/` (rungs.yaml, space.yaml, checks.yaml, endpoints.yaml
and the catalogue). Do not edit; change the declarations.

The ladder climbs to `freeze_design`; every rung may go only where this table says, and a stop goes to `feasibility`.

Loops still in the graph:

(none)

## The rungs

| rung | level | decided by | fixes | goes on to | budget | may ask |
|---|---|---|---|---|---|---|
| load | load | code: always |  | seed · stop |  |  |
| seed | seed | code: always |  | mechanism · stop |  |  |
| mechanism | mechanism | judgement: always |  | comparison · stop | 3 |  |
| comparison | comparison | judgement: always |  | controls · stop | 3 |  |
| controls | controls | code: no admitted covariate has safe timing and variation after fixed effects; judgement: admissible within-panel covariates require classification | modifiers | probe · stop | 3 |  |
| probe | probe | code: no actionable load-bearing proposal remains; judgement: a control, anticipation or conditional-trends proposal can change the design |  | rivals · stop | 3 |  |
| rivals | rivals | code: always | endpoint | check_design · stop |  |  |
| check_design | check design | code: always |  | pick_estimator · stop |  |  |
| pick_estimator | pick estimator | judgement: always |  | freeze_design · stop | 0 |  |
| freeze_design | freeze design | code: always |  | execute · stop |  |  |
| execute | execute | code: always |  | interpret · stop |  |  |
| interpret | interpret | judgement: always |  | figures · stop | 0 |  |

## The space

| dimension | values | decided by | read from | not run |
|---|---|---|---|---|
| endpoint | paired_two_wave_att, common_adoption_projection_effect, untreated_imputation_att, untreated_prefix_imputation_att, never_cohort_cell_att, clean_projection_att | rivals |  |  |
| modifiers | 0 to 3 | controls |  |  |

## Executable endpoint catalogue

Valid designs are conditional on the stated assumptions. Diagnostics cannot prove those assumptions.

| endpoint | target | configuration | trajectory | estimators / inference | terminal failures |
|---|---|---|---|---|---|
| paired_two_wave_att | att | {'shape': 'wide', 'comparison': 'untreated'} | load → seed → mechanism → comparison → controls → probe → rivals → check_design → pick_estimator → freeze_design → execute → interpret | twfe_static / row_robust, cluster_crv1, wild_webb, wild_rademacher, inference_unavailable | context_deficiency, unidentified_effect, unsupported_method, inadequate_support, exhausted_reasoning, execution_failure |
| common_adoption_projection_effect | att, att_horizon, att_horizon_profile | {'shape': 'long', 'cohorts': 1} | load → seed → mechanism → comparison → controls → probe → rivals → check_design → pick_estimator → freeze_design → execute → interpret | twfe_static, twfe_dynamic / cluster_crv1, cluster_crv3, wild_webb, wild_rademacher, inference_unavailable | context_deficiency, unidentified_effect, unsupported_method, inadequate_support, exhausted_reasoning, execution_failure |
| untreated_imputation_att | att, att_horizon, att_horizon_profile, att_cohort, att_cohort_profile | {'shape': 'long', 'comparison': 'untreated'} | load → seed → mechanism → comparison → controls → probe → rivals → check_design → pick_estimator → freeze_design → execute → interpret | did2s, did2s_dynamic, did2s_cohort / native_gmm, inference_unavailable | context_deficiency, unidentified_effect, unsupported_method, inadequate_support, exhausted_reasoning, execution_failure |
| untreated_prefix_imputation_att | att, att_horizon, att_horizon_profile, att_cohort, att_cohort_profile | {'shape': 'long', 'support': 'pre_last_adoption_prefix'} | load → seed → mechanism → comparison → controls → probe → rivals → check_design → pick_estimator → freeze_design → execute → interpret | did2s, did2s_dynamic, did2s_cohort / native_gmm, inference_unavailable | context_deficiency, unidentified_effect, unsupported_method, inadequate_support, exhausted_reasoning, execution_failure |
| never_cohort_cell_att | att, att_horizon, att_horizon_profile, att_cohort, att_cohort_profile, att_cohort_horizon | {'shape': 'long', 'comparison': 'never_treated'} | load → seed → mechanism → comparison → controls → probe → rivals → check_design → pick_estimator → freeze_design → execute → interpret | saturated_event_study / native_crv1, inference_unavailable | context_deficiency, unidentified_effect, unsupported_method, inadequate_support, exhausted_reasoning, execution_failure |
| clean_projection_att | att, att_horizon, att_horizon_profile | {'shape': 'long', 'comparison': 'clean_through_horizon'} | load → seed → mechanism → comparison → controls → probe → rivals → check_design → pick_estimator → freeze_design → execute → interpret | lpdid / native_crv1, native_crv3, inference_unavailable | context_deficiency, unidentified_effect, unsupported_method, inadequate_support, exhausted_reasoning, execution_failure |

### Context for `paired_two_wave_att`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| parallel_account | claim:trend_continues.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [True] |
| spillover_account | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| treated_group | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cluster | claim:assignment.level_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_population | claim:target.population | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated'] |
| layout | claim:grain.layout | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['wide'] |
| before_measurement | claim:grain.before_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| after_measurement | claim:grain.after_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| comparison_support | claim:assignment.never_treated_exists | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| target_averaging_policy | claim:target.weighting | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated_cell_shares', 'residualized_treatment_projection', 'linear_response_horizon_projection', 'clean_adopter_projection'] |

Assumptions: parallel trends for the declared comparison population; no unsupported anticipation or spillover.

Checks: shape, support, cluster, rank, pre_trends.

### Context for `common_adoption_projection_effect`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| parallel_account | claim:trend_continues.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [True] |
| spillover_account | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| treated_group | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cluster | claim:assignment.level_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| unit_keys | claim:grain.key_columns | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| clock | claim:change.date_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_population | claim:target.population | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated'] |
| layout | claim:grain.layout | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['long'] |
| comparison_support | claim:assignment.never_treated_exists | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| target_averaging_policy | claim:target.weighting | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated_cell_shares', 'residualized_treatment_projection', 'linear_response_horizon_projection', 'clean_adopter_projection'] |

Assumptions: parallel trends for the declared comparison population; no unsupported anticipation or spillover.

Checks: shape, support, cluster, rank, pre_trends.

### Context for `untreated_imputation_att`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| parallel_account | claim:trend_continues.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [True] |
| spillover_account | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| treated_group | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cluster | claim:assignment.level_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| unit_keys | claim:grain.key_columns | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| clock | claim:change.date_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_population | claim:target.population | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated'] |
| layout | claim:grain.layout | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['long'] |
| comparison_support | claim:assignment.never_treated_exists | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| target_averaging_policy | claim:target.weighting | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated_cell_shares', 'residualized_treatment_projection', 'linear_response_horizon_projection', 'clean_adopter_projection'] |

Assumptions: parallel trends for the declared comparison population; no unsupported anticipation or spillover.

Checks: shape, support, cluster, rank, pre_trends.

### Context for `untreated_prefix_imputation_att`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| parallel_account | claim:trend_continues.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [True] |
| spillover_account | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| treated_group | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cluster | claim:assignment.level_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| unit_keys | claim:grain.key_columns | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| clock | claim:change.date_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_population | claim:target.population | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated'] |
| layout | claim:grain.layout | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['long'] |
| prefix_scope | claim:target.prefix_scope_accepted | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| comparison_support | claim:assignment.never_treated_exists | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [False] |
| target_averaging_policy | claim:target.weighting | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated_cell_shares', 'residualized_treatment_projection', 'linear_response_horizon_projection', 'clean_adopter_projection'] |

Assumptions: parallel trends for the declared comparison population; no unsupported anticipation or spillover.

Checks: shape, support, cluster, rank, pre_trends.

### Context for `never_cohort_cell_att`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| parallel_account | claim:trend_continues.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [True] |
| spillover_account | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| treated_group | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cluster | claim:assignment.level_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| unit_keys | claim:grain.key_columns | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| clock | claim:change.date_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_population | claim:target.population | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated'] |
| layout | claim:grain.layout | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['long'] |
| comparison_support | claim:assignment.never_treated_exists | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| target_averaging_policy | claim:target.weighting | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated_cell_shares', 'residualized_treatment_projection', 'linear_response_horizon_projection', 'clean_adopter_projection'] |

Assumptions: parallel trends for the declared comparison population; no unsupported anticipation or spillover.

Checks: shape, support, cluster, rank, pre_trends.

### Context for `clean_projection_att`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| parallel_account | claim:trend_continues.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [True] |
| spillover_account | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| treated_group | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cluster | claim:assignment.level_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| unit_keys | claim:grain.key_columns | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| clock | claim:change.date_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_population | claim:target.population | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated'] |
| layout | claim:grain.layout | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['long'] |
| comparison_support | claim:assignment.never_treated_exists | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| accepted_clean_prefix | claim:target.prefix_scope_accepted | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {'claim:assignment.never_treated_exists': [False]} | [True] |
| target_averaging_policy | claim:target.weighting | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['treated_cell_shares', 'residualized_treatment_projection', 'linear_response_horizon_projection', 'clean_adopter_projection'] |

Assumptions: parallel trends for the declared comparison population; no unsupported anticipation or spillover.

Checks: shape, support, cluster, rank, pre_trends.

## The end states

Derived from the declarations: 6 engine configurations, 0 questions, 11 stops.

### Answers: the cells a design can reach

| cell | estimators | refuters |
|---|---|---|
| endpoint=paired_two_wave_att, shape=wide, comparison=untreated | twfe_static |  |
| endpoint=common_adoption_projection_effect, shape=long, cohorts=1 | twfe_static, twfe_dynamic |  |
| endpoint=untreated_imputation_att, shape=long, comparison=untreated | did2s, did2s_dynamic, did2s_cohort |  |
| endpoint=untreated_prefix_imputation_att, shape=long, support=pre_last_adoption_prefix | did2s, did2s_dynamic, did2s_cohort |  |
| endpoint=never_cohort_cell_att, shape=long, comparison=never_treated | saturated_event_study |  |
| endpoint=clean_projection_att, shape=long, comparison=clean_through_horizon | lpdid |  |

### Questions back

(none)

### Typed stops

| rung | cause | reason | what would fix it |
|---|---|---|---|
| load | intake | the hand-off names no file, or a column it names is not in the file | a hand-off whose file and columns exist |
| load | shape | the table fails a shape rule the lane declares | a table the rule holds on |
| mechanism | gate | the record was refused on every try | clearer notes on what the rung reads |
| comparison | gate | the record was refused on every try | clearer notes on what the rung reads |
| controls | space:modifiers | no value of modifiers the lane runs is left | a design inside the space |
| controls | gate | the record was refused on every try | clearer notes on what the rung reads |
| probe | gate | the record was refused on every try | clearer notes on what the rung reads |
| check_design | check:pre_trends.p_value | whether the treated and comparison groups were already moving differently before adoption; p value past 0.01 | a design that passes the check |
| pick_estimator | gate | the record was refused on every try | clearer notes on what the rung reads |
| execute | engine | the estimator failed to fit | a different estimator entry |
| interpret | gate | the record was refused on every try | clearer notes on what the rung reads |
