# causal-engine

Eight causal estimation methods, each checked against a real dataset and, where
one exists, a published number.

## Causal Studio MVP

The Streamlit studio accepts CSV, TSV, Excel, or Parquet uploads, profiles the
data, inventories multi-file Kaggle bundles, runs a bounded preparation ReAct
investigator, asks the uploader to confirm its context and repair plan, exposes
interactive general and lane-specific EDA, exposes all eight analysis designs,
runs lane-specific preflight, freezes a human-reviewed versioned design contract,
runs deterministic sensitivity checks, pauses for human publication review when
policy requires it, and emits an executable notebook audit bundle. Meaningful UI
decisions are recorded as sanitized, chained server-side events; ordinary chart
interaction does not call Vertex AI. Every approved repair/cohort combination
creates a content-addressed prepared-data version; the design contract is bound
to that exact version before an estimator can execute.

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

LangGraph is confined to checkpointed execution and the human publication
interrupt. Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and optionally
`LANGSMITH_PROJECT=causal-studio` to send graph traces to LangSmith. The app
works without LangSmith and without an LLM credential; without a configured
Vertex AI identity, the preparation investigator follows a deterministic
fallback over the same tool contracts and records that failure mode in the
trace. The preparation agent never calls the Gemini Developer API.

Configure Vertex AI with Application Default Credentials and the official GCP
environment names:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=true
export PREPARATION_MODEL=gemini-2.5-flash
```

See [`docs/MVP_ARCHITECTURE.md`](docs/MVP_ARCHITECTURE.md) for the boundaries,
policy rules, escalation points, and one-hour tradeoffs.

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
and no external artifact registry. Each is a real feature. Each gets added when
a test needs it, not in advance.

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
causal/studio_prep.py       upload profiling and approved repairs
causal/studio_eda.py        cached EDA summaries, cohort filters, chart frames
causal/studio_protocols.py  eight preflight/postflight protocols and contracts
causal/preparation_agent.py bounded ReAct investigation and tool contracts
causal/prompt_registry.py   immutable prompt versions and hashes
causal/monitoring.py        run metrics, alerts, and server interaction events
causal/studio_policy.py     executable allow/review/block rules
causal/studio_workflow.py   LangGraph estimate/check/policy/approval spine
causal/studio_export.py     executable notebook audit bundle
streamlit_app.py            human control surface
prompts/                    versioned, immutable agent prompts
fixtures.py          each lane paired with its dataset and benchmark
verify_lanes.py      the runner
data/                eight real CSVs, committed
```

Datasets are committed deliberately. The previous project gitignored them, so a
fresh clone could not run a single analysis.
