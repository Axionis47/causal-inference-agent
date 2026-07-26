# causal-engine

Eight causal estimation methods, each checked against a real dataset and, where
one exists, a published number.

```bash
pip install -r requirements.txt
python verify_lanes.py
```

Exit 0 means every lane ran and every benchmark held.

## What each lane is proved against

| lane | dataset | benchmark |
|---|---|---|
| `observational` | IHDP (747) | true ATE 4.0161, computed from the shipped `mu0`/`mu1` |
| `matching` | LaLonde NSW/PSID (614) | published range 1,000–2,200; the naive difference is **−635** |
| `iv` | Card 1995 (3,010) | published 2SLS return to schooling ≈ 0.13 |
| `survival` | heart failure (299) | hazard ratio 1.5455777143424887, pinned |
| `did` | Card & Krueger 1994 (794) | published DiD ≈ +2.76 FTE |
| `rdd` | bank recovery (1,882) | no published effect; checks the jump at 1,000 is found |
| `mediation` | student grades (649) | no published effect; sane output only |
| `time_series` | daily visitors (2,167) | no known intervention; sane output only |

**LaLonde is the load-bearing case.** Its naive treated-minus-control difference
is *negative*. A matching lane that genuinely adjusts pulls it to roughly
+1,100; one that quietly returns a raw difference cannot. That single check
catches the failure mode that matters most.

**Survival is a cross-implementation pin.** 1.5455777143424887 came from a
previous, separately written engine. This code reproduces it to zero
difference, having been written without reading that implementation.

## Why a script and not pytest

Every expected value here is a published figure or a truth computed from the
data. None was recorded from a previous run of this code, so none can go stale
— which is exactly how the previous project's tests rotted, asserting old
shapes and passing anyway.

`fixtures.cases()` returns plain data, so wrapping it in a parametrized pytest
is six lines whenever CI wants one.

## What v1 leaves out, on purpose

No inverse-probability weighting, no bootstrap standard errors, no collinearity
pruning, no one-hot encoding of categorical covariates (pass numeric columns),
no plots, no artifact registry. Each is a real feature. Each gets added when a
test needs it, not in advance.

## Known gaps

- **Matching standard errors run optimistic.** They ignore uncertainty in the
  propensity model, so that interval is narrower than it should be and spans
  zero. The point estimate is the benchmark claim, not the interval.
- **`rdd` prints "Mass points detected in the running variable."** That comes
  from `rdrobust`, not this code. The bank data has rounded amounts.
- **`matching` emits one `DeprecationWarning`**, from scikit-learn calling a
  scipy option that scipy 1.18 will drop. Upstream, not fixable here. It is
  counted in the `warns` column rather than suppressed.
- **RDD bandwidth is chosen by `rdrobust` and the estimate is noisy.** More data
  makes it worse, not better: RDD is local, so a larger sample narrows the
  window rather than sharpening the estimate.

## Layout

```
causal/estimate.py   the Estimate dataclass and LaneError
causal/prep.py       the four checks every lane repeats
causal/lanes.py      the eight methods, one file, top to bottom
fixtures.py          each lane paired with its dataset and benchmark
verify_lanes.py      the runner
data/                eight real CSVs, committed
```

Datasets are committed deliberately. The previous project gitignored them, so a
fresh clone could not run a single analysis.
