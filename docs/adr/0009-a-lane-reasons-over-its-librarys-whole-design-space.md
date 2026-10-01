# 9. A lane reasons over its library's whole design space

ADR 0008 stands: a judgement is a bounded episode and the context is a ladder. This decision says what the ladder must
reach. It was made when an audit found the two design lanes thinner than the adjustment lane: their identifying rung
judged blind, their tools were written for a cross-section, and each used a sliver of its library, with catalogue
entries that no code could run.

## The principle, stated once

A causal design is an argument that a comparison in the data equals the counterfactual the question asks about. Every
step of that argument depends on the steps before it and on nothing after it. Read in that order the argument is the
same ten levels in every lane: what is compared; how assignment happened; the structure that carries identification;
the evidence for the identifying assumption; the identifying judgement; what else touches the outcome; where the
effect differs; the risks of this design; how to compute it; freeze, run, attack, answer. Each lane climbs these levels
as its rungs, in this order, and a rung reads only the rungs below it.

A design is a point in the space its library spans, not a named recipe. **Every dimension of that space is decided by
exactly one of three things**: a fact code computes, a judgement the model makes in a bounded episode, or a declaration
in the lane's catalogue. The design-space tables in [lanes.md](../lanes.md) list every dimension the library offers and
its decider. Widening the space is a new option at an existing level, never a new rung.

## What follows from it

**Evidence rungs sit below judgement rungs.** The evidence for an identifying assumption is computed by code before the
judgement that rests on it, as addressed lines the judgement must cite: the pre-period paths, the leads test and the
composition before the comparison is judged; the density test and its binomial windows before the line; each
covariate's jump at the line before the covariates are placed. Nothing about the effect is in them. A verdict that
contradicts the evidence is refused by the gate, not corrected after the fact by a check.

**The outcome rule is derived per design, not copied.** In a panel the outcome by group before the change is evidence
for the assumption and reveals nothing of the effect, so the comparison rung may see it and nothing after the change.
At a cutoff the outcome by side is the effect at every step, so nothing sees it before the freeze. The rule is code in
the harness, a mask the lane passes.

**A judgement is where the literature gives guidance and not a number.** The width at a cutoff is a judgement, because
the Foundations say when a two-sided or a coverage-error width fits and leave the choice to the analyst; the kernel is
a declaration, because the literature says it matters little. The estimator among the survivors is a judgement; which
entries survive is a fact. How the treated group came to be chosen is a judgement where the pack leaves it open, and a
fact where it does not.

**The catalogue is keyed on facts, and no yaml is dead.** An entry applies by one rule over facts the lane computes;
every yaml key is a field of its entry model; a test per lane asserts both. An inference entry, a falsification, a
sensitivity runs because its facts hold, never because something picked it. A falsification carries a verdict under
one declared pass rule; a sensitivity reports a range and no verdict.

**The lane hands back the risks of its own design by code**, from the mechanism and the rungs below: a group chosen for
its trend, a stated lead, few treated units; manipulation at the line, a discrete score, one-sided take-up, a thin
side. A judgement rung may add the risks only the story can raise, from a fixed list and cited.

## What it costs

More library calls before the freeze: the density test, a fit per covariate, every width selector on both sides, the
leads fit, run once and read from the ladder by the checks. One more episode per lane (the window; the mechanism). A
longer test suite, because each falsification is proved on a clean panel and on one doctored to break what it tests.
Nothing changes in the adjustment lane; if the evidence rungs prove right in a real run, giving it one (balance and
overlap before the roles) is the next decision.
