# The discontinuity lane's ladder

Written by `make ladder` from `causal_agent/families/discontinuity/lane/knowledge/` (rungs.yaml, space.yaml, checks.yaml, endpoints.yaml
and the catalogue). Do not edit; change the declarations.

The ladder climbs to `freeze_designs`; every rung may go only where this table says, and a stop goes to `feasibility`.

Loops still in the graph:

(none)

## The rungs

| rung | level | decided by | fixes | goes on to | budget | may ask |
|---|---|---|---|---|---|---|
| load | load | code: always | dependence | validate_context · stop |  |  |
| validate_context | validate context | code: always |  | seed · stop |  |  |
| seed | seed | code: always |  | score |  |  |
| score | score | code: always |  | shape_table · stop |  |  |
| shape_table | shape table | code: always | kind, support_regime | covariates · stop |  |  |
| covariates | covariates | judgement: always |  | density · stop | 4 |  |
| density | density | code: always |  | balance |  |  |
| balance | balance | code: always |  | line |  |  |
| line | line | judgement: always |  | assumptions · stop | 3 |  |
| assumptions | assumptions | judgement: always | framework | rivals · stop | 0 |  |
| rivals | rivals | code: always | estimand | check_candidates |  |  |
| check_candidates | check candidates | code: always |  | freeze_designs |  |  |
| freeze_designs | freeze designs | code: always |  | execute · stop |  |  |
| execute (worker) | execute | code: always |  | interpret |  |  |
| interpret | interpret | judgement: always |  | figures | 0 |  |

## The space

| dimension | values | decided by | read from | not run |
|---|---|---|---|---|
| framework | continuity, local_randomisation | assumptions |  |  |
| kind | sharp, fuzzy | shape_table |  |  |
| estimand | effect_at_cutoff, complier_effect_at_cutoff, itt_at_cutoff, effect_in_window, complier_effect_in_window, itt_in_window | rivals |  |  |
| support_regime | dense, coarse | shape_table |  |  |
| dependence | independent, clustered | load |  |  |
| framework=local_randomisation+dependence=clustered |  |  |  | the installed randomisation engine does not permute assignment clusters |

## Executable endpoint catalogue

Valid designs are conditional on the stated assumptions. Diagnostics cannot prove those assumptions.

| endpoint | target | configuration | trajectory | estimators / inference | terminal failures |
|---|---|---|---|---|---|
| sharp_local_polynomial | effect_at_cutoff | {'framework': 'continuity', 'kind': 'sharp', 'estimand': 'effect_at_cutoff'} | load → validate_context → seed → score → shape_table → covariates → density → balance → line → assumptions → rivals → check_candidates → freeze_designs | local_linear / robust_bc, cluster_entity, cluster_few | context_deficiency, invalid_rule, insufficient_support, assumptions_rejected, engine_failure |
| fuzzy_local_polynomial | complier_effect_at_cutoff | {'framework': 'continuity', 'kind': 'fuzzy', 'estimand': 'complier_effect_at_cutoff'} | load → validate_context → seed → score → shape_table → covariates → density → balance → line → assumptions → rivals → check_candidates → freeze_designs | local_linear_fuzzy / robust_bc, cluster_entity, cluster_few | context_deficiency, invalid_rule, insufficient_support, assumptions_rejected, engine_failure |
| eligibility_local_polynomial | itt_at_cutoff | {'framework': 'continuity', 'kind': 'fuzzy', 'estimand': 'itt_at_cutoff'} | load → validate_context → seed → score → shape_table → covariates → density → balance → line → assumptions → rivals → check_candidates → freeze_designs | local_linear_itt / robust_bc, cluster_entity, cluster_few | context_deficiency, invalid_rule, insufficient_support, assumptions_rejected, engine_failure |
| sharp_local_randomisation | effect_in_window | {'framework': 'local_randomisation', 'kind': 'sharp', 'estimand': 'effect_in_window'} | load → validate_context → seed → score → shape_table → covariates → density → balance → line → assumptions → rivals → check_candidates → freeze_designs | local_randomisation / randomisation | context_deficiency, invalid_rule, insufficient_support, assumptions_rejected, engine_failure, dependent_assignment |
| fuzzy_local_randomisation | complier_effect_in_window | {'framework': 'local_randomisation', 'kind': 'fuzzy', 'estimand': 'complier_effect_in_window'} | load → validate_context → seed → score → shape_table → covariates → density → balance → line → assumptions → rivals → check_candidates → freeze_designs | local_randomisation_fuzzy / randomisation | context_deficiency, invalid_rule, insufficient_support, assumptions_rejected, engine_failure, dependent_assignment |
| eligibility_local_randomisation | itt_in_window | {'framework': 'local_randomisation', 'kind': 'fuzzy', 'estimand': 'itt_in_window'} | load → validate_context → seed → score → shape_table → covariates → density → balance → line → assumptions → rivals → check_candidates → freeze_designs | local_randomisation_itt / randomisation | context_deficiency, invalid_rule, insufficient_support, assumptions_rejected, engine_failure, dependent_assignment |

### Context for `sharp_local_polynomial`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| target_locality | claim:target.locality | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['cutoff', 'window'] |
| rule | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| score | claim:assignment.score_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cutoff | claim:assignment.cutoff | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| side | claim:assignment.treated_side | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| equality | claim:assignment.cutoff_value_treated | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<score>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| movability | claim:assignment.movable | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| other_changes | claim:cutoff_only.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| compliance | claim:assignment.compliance | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['sharp'] |

Assumptions: no coincident intervention confounds the contrast; no selective score manipulation; potential-outcome regression functions are continuous at the cutoff.

Checks: sides, support, density, covariate_continuity.

### Context for `fuzzy_local_polynomial`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| target_locality | claim:target.locality | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['cutoff', 'window'] |
| target_comparison | claim:target.comparison | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {'claim:assignment.compliance': ['fuzzy']} | ['receipt', 'eligibility'] |
| rule | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| score | claim:assignment.score_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cutoff | claim:assignment.cutoff | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| side | claim:assignment.treated_side | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| equality | claim:assignment.cutoff_value_treated | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<score>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| movability | claim:assignment.movable | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| other_changes | claim:cutoff_only.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| receipt | claim:assignment.treatment_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| receipt_level | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exclusion | claim:exclusion.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| monotonicity | claim:monotonicity.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| compliance | claim:assignment.compliance | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['fuzzy'] |

Assumptions: no coincident intervention confounds the contrast; no selective score manipulation; potential-outcome regression functions are continuous at the cutoff; eligibility affects outcome only through receipt; eligibility moves receipt in one direction.

Checks: sides, support, density, covariate_continuity, first_stage.

### Context for `eligibility_local_polynomial`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| target_locality | claim:target.locality | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['cutoff', 'window'] |
| target_comparison | claim:target.comparison | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {'claim:assignment.compliance': ['fuzzy']} | ['receipt', 'eligibility'] |
| rule | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| score | claim:assignment.score_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cutoff | claim:assignment.cutoff | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| side | claim:assignment.treated_side | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| equality | claim:assignment.cutoff_value_treated | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<score>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| movability | claim:assignment.movable | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| other_changes | claim:cutoff_only.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| compliance | claim:assignment.compliance | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['fuzzy'] |

Assumptions: no coincident intervention confounds the contrast; no selective score manipulation; potential-outcome regression functions are continuous at the cutoff.

Checks: sides, support, density, covariate_continuity.

### Context for `sharp_local_randomisation`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| target_locality | claim:target.locality | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['cutoff', 'window'] |
| rule | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| score | claim:assignment.score_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cutoff | claim:assignment.cutoff | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| side | claim:assignment.treated_side | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| equality | claim:assignment.cutoff_value_treated | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<score>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| movability | claim:assignment.movable | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| other_changes | claim:cutoff_only.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| randomisation_account | claim:local_randomisation.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| compliance | claim:assignment.compliance | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['sharp'] |

Assumptions: no coincident intervention confounds the contrast; no selective score manipulation; assignment inside the frozen window is as-if complete randomisation; the additive-effect null is appropriate for randomisation test inversion.

Checks: sides, support, density, covariate_continuity.

### Context for `fuzzy_local_randomisation`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| target_locality | claim:target.locality | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['cutoff', 'window'] |
| target_comparison | claim:target.comparison | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {'claim:assignment.compliance': ['fuzzy']} | ['receipt', 'eligibility'] |
| rule | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| score | claim:assignment.score_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cutoff | claim:assignment.cutoff | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| side | claim:assignment.treated_side | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| equality | claim:assignment.cutoff_value_treated | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<score>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| movability | claim:assignment.movable | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| other_changes | claim:cutoff_only.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| randomisation_account | claim:local_randomisation.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| receipt | claim:assignment.treatment_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| receipt_level | claim:assignment.treated_level | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exclusion | claim:exclusion.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| monotonicity | claim:monotonicity.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| compliance | claim:assignment.compliance | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['fuzzy'] |

Assumptions: no coincident intervention confounds the contrast; no selective score manipulation; assignment inside the frozen window is as-if complete randomisation; the additive-effect null is appropriate for randomisation test inversion; eligibility affects outcome only through receipt; eligibility moves receipt in one direction.

Checks: sides, support, density, covariate_continuity, first_stage.

### Context for `eligibility_local_randomisation`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| target_locality | claim:target.locality | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['cutoff', 'window'] |
| target_comparison | claim:target.comparison | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {'claim:assignment.compliance': ['fuzzy']} | ['receipt', 'eligibility'] |
| rule | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| score | claim:assignment.score_column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| cutoff | claim:assignment.cutoff | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| side | claim:assignment.treated_side | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| equality | claim:assignment.cutoff_value_treated | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<score>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| movability | claim:assignment.movable | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| other_changes | claim:cutoff_only.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| unit | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| randomisation_account | claim:local_randomisation.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |
| compliance | claim:assignment.compliance | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | ['fuzzy'] |

Assumptions: no coincident intervention confounds the contrast; no selective score manipulation; assignment inside the frozen window is as-if complete randomisation; the additive-effect null is appropriate for randomisation test inversion.

Checks: sides, support, density, covariate_continuity.

## The end states

Derived from the declarations: 6 engine configurations, 0 questions, 6 stops.

### Answers: the cells a design can reach

| cell | estimators | refuters |
|---|---|---|
| endpoint=sharp_local_polynomial, framework=continuity, kind=sharp, estimand=effect_at_cutoff | local_linear |  |
| endpoint=fuzzy_local_polynomial, framework=continuity, kind=fuzzy, estimand=complier_effect_at_cutoff | local_linear_fuzzy |  |
| endpoint=eligibility_local_polynomial, framework=continuity, kind=fuzzy, estimand=itt_at_cutoff | local_linear_itt |  |
| endpoint=sharp_local_randomisation, framework=local_randomisation, kind=sharp, estimand=effect_in_window | local_randomisation |  |
| endpoint=fuzzy_local_randomisation, framework=local_randomisation, kind=fuzzy, estimand=complier_effect_in_window | local_randomisation_fuzzy |  |
| endpoint=eligibility_local_randomisation, framework=local_randomisation, kind=fuzzy, estimand=itt_in_window | local_randomisation_itt |  |

### Questions back

(none)

### Typed stops

| rung | cause | reason | what would fix it |
|---|---|---|---|
| load | intake | the hand-off names no file, or a column it names is not in the file | a hand-off whose file and columns exist |
| load | shape | the table fails a shape rule the lane declares | a table the rule holds on |
| covariates | gate | the record was refused on every try | clearer notes on what the rung reads |
| line | gate | the record was refused on every try | clearer notes on what the rung reads |
| assumptions | space:framework=local_randomisation+dependence=clustered | the installed randomisation engine does not permute assignment clusters | a design inside the space |
| assumptions | gate | the record was refused on every try | clearer notes on what the rung reads |
