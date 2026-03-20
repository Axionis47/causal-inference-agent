# Skill: DAG Review

Load when: reviewing causal graph structures, adjustment sets, or identification strategy.

## What a valid DAG must have

1. **Treatment and outcome identified**: Clear T -> Y path (or absence of one)
2. **No cycles**: Directed acyclic graph. If cycles exist, the model is wrong.
3. **Confounders on backdoor paths**: Every non-causal path from T to Y must be blockable
4. **No colliders conditioned on**: Conditioning on a collider opens a spurious path

## Backdoor criterion

A set Z satisfies the backdoor criterion relative to (T, Y) if:
1. No node in Z is a descendant of T
2. Z blocks every path between T and Y that contains an arrow into T

If Z satisfies backdoor, adjusting for Z gives the causal effect of T on Y.

## Common DAG mistakes to catch

| Mistake | Why it's wrong | What to do |
|---------|---------------|------------|
| Conditioning on a mediator | Blocks part of the causal effect | Remove mediators from adjustment set |
| Conditioning on a collider | Opens spurious associations | Remove colliders from adjustment set |
| Missing confounder | Backdoor path unblocked, biased estimate | Flag as limitation, run sensitivity analysis |
| Reversed edge direction | Wrong causal direction assumed | Check temporal ordering, domain knowledge |
| Including post-treatment variables | Introduces post-treatment bias | Only adjust for pre-treatment variables |

## Variable classification

Every variable in the DAG should be classified:

| Role | Definition | Adjust for it? |
|------|-----------|---------------|
| Confounder | Causes both T and Y | Yes — must adjust |
| Mediator | On the causal path T -> M -> Y | No — blocks the effect |
| Collider | Caused by both T and Y (or their ancestors) | No — opens spurious path |
| Instrument | Causes T, affects Y only through T | No — use for IV estimation |
| Immutable | Cannot be caused by treatment (age, sex) | Safe to adjust |
| Irrelevant | No connection to T or Y | Omit — adds noise |

## Review checklist for DAGExpertAgent output

- [ ] Treatment and outcome nodes correctly identified
- [ ] All confounder edges are plausible (domain knowledge check)
- [ ] No post-treatment variables in adjustment set
- [ ] No colliders conditioned on
- [ ] No mediators in adjustment set (unless total effect is not the goal)
- [ ] Temporal ordering respected (causes precede effects)
- [ ] Immutable variables (demographics) correctly marked
- [ ] Adjustment set is minimal sufficient (not over-adjusted)

## When DAG discovery fails

If no DAG is produced (algorithms disagree or data is insufficient):
1. Fall back to domain knowledge for confounder selection
2. Use the ConfounderDiscoveryAgent's statistical approach
3. Run sensitivity analysis (E-value) to bound unmeasured confounding
4. Document the limitation in the notebook
