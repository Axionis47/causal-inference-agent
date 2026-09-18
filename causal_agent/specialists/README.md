# specialists

One subgraph per family. Each receives a hand-off (family, outcome, treatment, scope, relevant columns) and turns it into a runnable analysis in its own library's vocabulary. A specialist never decides whether it applies; the router did.

- `dowhy/` — the adjustment family, built. Contrasts, a cited graph, DoWhy identification, design checks, a frozen Design, estimate, refute, interpret. See its README.
- `did/` — the diff_in_diff family, built on pyfixest. Groups, periods, a canonical panel, controls, pre-trend and placebo checks, a frozen Design, fit, interpret. See its README.
- `rd/` — the discontinuity family, built on rdrobust and rddensity. Score and cutoff, a canonical cutoff table, predetermined covariates, density, first-stage and continuity checks, a frozen Design, local polynomial fit, placebo cutoffs, bandwidth grid and donuts, interpret. See its README.
- every other family — a stub that reports the hand-off and says the specialist is not built yet.

Each specialist lives in its own folder with its code, tests, evals, and knowledge files, and is looked up by family name in `SPECIALISTS`. To add one: build the subgraph, add its factory to `BUILT` in `__init__.py`, flip the family's `status` in `knowledge/families.yaml`.
