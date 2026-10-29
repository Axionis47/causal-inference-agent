# The adjustment lane's ladder

Written by `make ladder` from `causal_agent/families/adjustment/lane/knowledge/` (rungs.yaml, space.yaml, checks.yaml, endpoints.yaml
and the catalogue). Do not edit; change the declarations.

The ladder climbs to `freeze_design`; every rung may go only where this table says, and a stop goes to `feasibility`.

Loops still in the graph:

- rivals → relate, at most 1: every candidate graph is invalid before any outcome effect is seen

## The rungs

| rung | level | decided by | fixes | goes on to | budget | may ask |
|---|---|---|---|---|---|---|
| load | admitted context and table | code: always | outcome | case · stop |  |  |
| case | evidence classifications | code: always |  | pair |  |  |
| pair | treatment comparisons | pack: observed confirmed treatment comparison; judgement: otherwise | treatment, contrasts | seed · stop | 2 |  |
| seed | immutable source assertions | code: always |  | mechanism |  |  |
| mechanism | assignment mechanism | judgement: always |  | relate · stop | 4 |  |
| relate | relationship hypotheses | judgement: always |  | probe · stop | 8 |  |
| probe | load-bearing uncertainty | judgement: unresolved relationships only; code: no uncertainty |  | rivals | 4 |  |
| rivals | identification endpoints | code: always | endpoint, modifiers | relate, check_design · stop |  |  |
| check_design | endpoint-specific checks | code: always |  | pick_estimator · stop |  |  |
| pick_estimator | eligible estimator per design | code: exactly one eligible estimator for a design; judgement: several eligible estimators for a design |  | freeze_design · stop | 3 |  |
| freeze_design | fixed execution inputs | code: always |  | analyse |  |  |
| analyse (worker) | estimation and diagnostics | code: always |  | after_analyse |  |  |
| after_analyse | execution status | code: always |  | interpret · stop |  |  |
| interpret | target-aware explanation | judgement: any successful estimate |  | figures | 0 |  |

## The space

| dimension | values | decided by | read from | not run |
|---|---|---|---|---|
| treatment | binary_contrast | pair |  | continuous: dose response is outside the declared adjustment endpoints; multiple: multiple simultaneous treatments are not supported |
| outcome | continuous, binary | load |  | other: outcome must be continuous numeric or explicitly encoded binary |
| endpoint | total.unadjusted, total.backdoor, total.general_adjustment, iv.wald_late, iv.linear_single, iv.linear_multiple, iv.linear_conditional, frontdoor.linear_total, mediation.linear_nie, mediation.linear_nde, adjustment.linear_conditional, structural.zero | rivals |  |  |
| contrasts | 1 to 11 | pair |  | above: more than twelve treatment levels requires a dose or multicategory design outside this lane |
| modifiers | 0 to 3 | rivals |  |  |
| endpoint=frontdoor.linear_total+outcome=binary |  |  |  | the frontdoor linear adapter requires continuous outcome |
| endpoint=mediation.linear_nie+outcome=binary |  |  |  | natural indirect effects support additive continuous models only |
| endpoint=mediation.linear_nde+outcome=binary |  |  |  | natural direct effects support additive continuous models only |
| endpoint=adjustment.linear_conditional+outcome=binary |  |  |  | conditional effects support continuous linear interaction models only |

## Executable endpoint catalogue

Valid designs are conditional on the stated assumptions. Diagnostics cannot prove those assumptions.

| endpoint | target | configuration | trajectory | estimators / inference | terminal failures |
|---|---|---|---|---|---|
| total.unadjusted | ate | {'road': 'backdoor', 'adjustment': 'empty'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | linear_regression, generalized_linear_model / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| total.backdoor | ate, att, atc | {'road': 'backdoor', 'adjustment': 'nonempty'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | linear_regression, generalized_linear_model, propensity_score_stratification, propensity_score_weighting, propensity_score_matching / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| total.general_adjustment | ate | {'road': 'general_adjustment'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | general_linear, general_glm / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| iv.wald_late | late | {'road': 'iv', 'instrument': 'binary', 'target': 'late'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | wald / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| iv.linear_single | ate | {'road': 'iv', 'instruments': 'one'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | instrumental_variable / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| iv.linear_multiple | ate | {'road': 'iv', 'instruments': 'multiple'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | iv_2sls / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| iv.linear_conditional | ate | {'road': 'iv', 'controls': 'nonempty'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | iv_2sls / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| frontdoor.linear_total | ate | {'road': 'frontdoor', 'mediators': 'one', 'outcome': 'continuous'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | frontdoor_two_stage / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| mediation.linear_nie | nie | {'road': 'mediation', 'target': 'nie', 'outcome': 'continuous'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | mediation_two_stage / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| mediation.linear_nde | nde | {'road': 'mediation', 'target': 'nde', 'outcome': 'continuous'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | mediation_two_stage / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| adjustment.linear_conditional | conditional | {'road': 'adjustment', 'target': 'conditional', 'outcome': 'continuous'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | linear_interaction, general_linear_interaction / bootstrap_100 | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| structural.zero | ate | {'road': 'zero'} | load → case → pair → seed → mechanism → relate → probe → rivals → check_design → pick_estimator → freeze_design → analyse → after_analyse → interpret | structural_zero / graph_implication | context_deficient, invalid_graph, nonidentified, unsupported_shape, support_failure, fit_failed |
| identification.id_expression | ate | {'road': 'id', 'observed_only': True} | rivals |  /  | unsupported_latent_representation, expression_only |

### Context for `total.unadjusted`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions.

Checks: arms, encoding, support, identification_certificate.

### Context for `total.backdoor`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions.

Checks: arms, encoding, support, identification_certificate.

### Context for `total.general_adjustment`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions.

Checks: arms, encoding, support, identification_certificate.

### Context for `iv.wald_late`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| instrument | claim:exclusion.column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exclusion | claim:exclusion.why | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exogenous | claim:exclusion.exogenous | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_assumption | claim:monotonicity.believed | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; exclusion; exogeneity; monotonicity.

Checks: arms, encoding, support, identification_certificate, first_stage, rank.

### Context for `iv.linear_single`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| instrument | claim:exclusion.column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exclusion | claim:exclusion.why | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exogenous | claim:exclusion.exogenous | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_assumption | claim:exclusion.homogeneous | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; exclusion; exogeneity; homogeneous_additive_effect.

Checks: arms, encoding, support, identification_certificate, first_stage, rank.

### Context for `iv.linear_multiple`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| instrument | claim:exclusion.column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exclusion | claim:exclusion.why | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exogenous | claim:exclusion.exogenous | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_assumption | claim:exclusion.homogeneous | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; exclusion; exogeneity; homogeneous_additive_effect.

Checks: arms, encoding, support, identification_certificate, first_stage, rank.

### Context for `iv.linear_conditional`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| instrument | claim:exclusion.column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exclusion | claim:exclusion.why | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| exogenous | claim:exclusion.exogenous | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| target_assumption | claim:exclusion.homogeneous | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| controls | claim:exclusion.controls | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; exclusion; conditional_exogeneity; homogeneous_additive_effect.

Checks: arms, encoding, support, identification_certificate, first_stage, rank.

### Context for `frontdoor.linear_total`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| mediator | claim:mediator.column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| mechanism | claim:mediator.why | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| linear | claim:mediator.linear_additive | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; linear_additive_stages; complete_mediation; frontdoor_criterion.

Checks: arms, encoding, support, identification_certificate, mediator_variation, stage_identification.

### Context for `mediation.linear_nie`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| mediator | claim:mediator.column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| mechanism | claim:mediator.why | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| linear | claim:mediator.linear_additive | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sequential | claim:mediator.sequential_ignorability | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| no_induced_confounder | claim:mediator.no_treatment_confounder | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; linear_additive_stages; sequential_ignorability; no_treatment_induced_mediator_outcome_confounding.

Checks: arms, encoding, support, identification_certificate, mediator_variation, stage_identification.

### Context for `mediation.linear_nde`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| mediator | claim:mediator.column | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| mechanism | claim:mediator.why | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| linear | claim:mediator.linear_additive | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sequential | claim:mediator.sequential_ignorability | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| no_induced_confounder | claim:mediator.no_treatment_confounder | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; linear_additive_stages; sequential_ignorability; no_treatment_induced_mediator_outcome_confounding.

Checks: arms, encoding, support, identification_certificate, mediator_variation, stage_identification.

### Context for `adjustment.linear_conditional`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| modifiers | col:<modifier>.may_modify | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| modifier_timing | col:<modifier>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions.

Checks: arms, encoding, support, identification_certificate.

### Context for `structural.zero`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| explicit_no_causal_path | claim:change.no_causal_path | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [True] |

Assumptions: consistency; no_interference; graph_relationships_are_explicit_assumptions; a supported absence of every directed treatment-outcome path, not merely an omitted edge.

Checks: arms, encoding, support, identification_certificate.

### Context for `identification.id_expression`

| requirement | address | accepted status | acceptable sources | permitted unknown | condition | allowed values |
|---|---|---|---|---|---|---|
| grain | claim:grain.row_is | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| assignment | claim:assignment.kind | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| sampling | claim:sampling.how | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| hidden | claim:unobserved.exists | confirmed | user:, doc:, data, code:, fixture:, eval: | True | {} | [] |
| spillover | claim:spillover.possible | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |
| timing | col:<outcome>.when | confirmed | user:, doc:, data, code:, fixture:, eval: | False | {} | [] |

Assumptions: fully_observed_DAG.

Checks: observable_expression.

## The end states

Derived from the declarations: 12 engine configurations, 0 questions, 22 stops.

### Answers: the cells a design can reach

| cell | estimators | refuters |
|---|---|---|
| endpoint=total.unadjusted, road=backdoor, adjustment=empty | linear_regression, generalized_linear_model |  |
| endpoint=total.backdoor, road=backdoor, adjustment=nonempty | linear_regression, generalized_linear_model, propensity_score_stratification, propensity_score_weighting, propensity_score_matching |  |
| endpoint=total.general_adjustment, road=general_adjustment | general_linear, general_glm |  |
| endpoint=iv.wald_late, road=iv, instrument=binary, target=late | wald |  |
| endpoint=iv.linear_single, road=iv, instruments=one | instrumental_variable |  |
| endpoint=iv.linear_multiple, road=iv, instruments=multiple | iv_2sls |  |
| endpoint=iv.linear_conditional, road=iv, controls=nonempty | iv_2sls |  |
| endpoint=frontdoor.linear_total, road=frontdoor, mediators=one, outcome=continuous | frontdoor_two_stage |  |
| endpoint=mediation.linear_nie, road=mediation, target=nie, outcome=continuous | mediation_two_stage |  |
| endpoint=mediation.linear_nde, road=mediation, target=nde, outcome=continuous | mediation_two_stage |  |
| endpoint=adjustment.linear_conditional, road=adjustment, target=conditional, outcome=continuous | linear_interaction, general_linear_interaction |  |
| endpoint=structural.zero, road=zero | structural_zero |  |

### Questions back

(none)

### Typed stops

| rung | cause | reason | what would fix it |
|---|---|---|---|
| load | space:outcome=other | outcome must be continuous numeric or explicitly encoded binary | a file inside the space |
| load | intake | the hand-off names no file, or a column it names is not in the file | a hand-off whose file and columns exist |
| load | shape | the table fails a shape rule the lane declares | a table the rule holds on |
| pair | space:treatment=continuous | dose response is outside the declared adjustment endpoints | a design inside the space |
| pair | space:treatment=multiple | multiple simultaneous treatments are not supported | a design inside the space |
| pair | space:contrasts | no value of contrasts the lane runs is left | a design inside the space |
| pair | space:contrasts=above | more than twelve treatment levels requires a dose or multicategory design outside this lane | a design inside the space |
| pair | gate | the record was refused on every try | clearer notes on what the rung reads |
| mechanism | gate | the record was refused on every try | clearer notes on what the rung reads |
| relate | gate | the record was refused on every try | clearer notes on what the rung reads |
| rivals | space:modifiers | no value of modifiers the lane runs is left | a design inside the space |
| rivals | space:endpoint=frontdoor.linear_total+outcome=binary | the frontdoor linear adapter requires continuous outcome | a design inside the space |
| rivals | space:endpoint=mediation.linear_nie+outcome=binary | natural indirect effects support additive continuous models only | a design inside the space |
| rivals | space:endpoint=mediation.linear_nde+outcome=binary | natural direct effects support additive continuous models only | a design inside the space |
| rivals | space:endpoint=adjustment.linear_conditional+outcome=binary | conditional effects support continuous linear interaction models only | a design inside the space |
| rivals | graph | what the rungs built fails verification | clearer column notes about what fed the decision |
| check_design | check:arms.min_per_arm | how many rows got the change and how many did not; min per arm past 20 | a design that passes the check |
| check_design | check:overlap.common_support_share | whether both arms are found across the whole range of what the offer depended on; common support share past 0.8 | a design that passes the check |
| check_design | check:overlap.max_auc | whether both arms are found across the whole range of what the offer depended on; max auc past 0.95 | a design that passes the check |
| check_design | check:balance.smd | how alike the two arms are on a column before the adjustment, and after weighting; smd past 0.25 | a design that passes the check |
| pick_estimator | gate | the record was refused on every try | clearer notes on what the rung reads |
| after_analyse | engine | the estimator failed to fit | a different estimator entry |
