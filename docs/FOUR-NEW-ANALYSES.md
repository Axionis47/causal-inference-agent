# Four public-data end-to-end analyses

The dataset manifest is [`evals/four-new-journeys.v1.json`](../evals/four-new-journeys.v1.json).
Each source is pinned by SHA-256. The execution command downloads the real source,
applies the declared projection, and sends the resulting CSV through the production
intake, design, preparation, estimation, and presentation runtime.

The intake adapter supplies frozen public-source metadata through the existing
dataset-client interface. It does not call Kaggle. Design, claim review, and figure
selection use the configured live Vertex model; no scripted model decisions or
precompiled designs are supplied.

| Method | Dataset | Defined analysis |
| --- | --- | --- |
| Randomized experiment | Rock the Vote | Advertising assignment and youth turnout across 85 randomized cable systems, with randomization strata |
| AIPW | NHEFS | Smoking cessation and weight change in the 1,566-person complete-case cohort, adjusting for nine baseline variables |
| Difference-in-differences | Castle Doctrine | Log homicide in 13 states adopting in 2006 versus 29 comparison states, excluding the transition year |
| Sharp regression discontinuity | Head Start | Local reduced-form effect of assistance eligibility at the poverty cutoff on child mortality |

## Run and resume

Load the existing local environment configuration before running these commands.
The runtime requires PostgreSQL, S3-compatible storage, Vertex credentials, and
LangSmith tracing. Do not commit credentials.

```bash
.venv/bin/python tools/four_analyses.py start \
  --fixtures evals/four-new-journeys.v1.json \
  --output-dir output/four-verified-20260909
```

Add `--case-id rock_the_vote_rct`, `nhefs_aipw`, `castle_doctrine_did`, or
`head_start_rdd` to run one case. Independent cases may run in parallel.

The command pauses at clarification and design approval. Review the exact request
under `outbox/<case_id>` and place its bound response under `inbox/<case_id>`.
The response must name the request hash, run, case, and actual reviewer. An absent
response never means approval. Resume with the same arguments and `resume` instead
of `start`. Use `status` to refresh the combined report.

## What is retained

Each case directory contains its original source bytes, projected input CSV,
source and input hashes, live model-call records, review history, every committed
artifact and envelope, terminal result, and readable report. Completed cases also
export the frozen summary and PNG/SVG figures. Exported bytes are checked against
the persisted bundle hashes before the case is marked passed.

Failed attempts remain separate under `attempts/`; they never count as delivered.
The four-case command is an execution check, not approval of the separate
human-gold release benchmark.

Execution success and final acceptance are separate. `execution_passed` checks
the stage artifacts, live decisions, approval and exported hashes. The aggregate
`passed` value additionally requires `acceptance.json`, binding the analysis ID,
input hash and presentation bundle hash to numerical, chart and interpretation
verification. Its reviewer identity and evidence paths record who checked what.
The per-case legacy `passed` field remains an execution check for compatibility.

The final acceptance batch uses a fresh output directory. Head Start's context
now includes the documented mortality unit and poverty percentage from the
[replication dictionary](https://users.ssc.wisc.edu/~bhansen/econometrics/LM2007_description.pdf)
and [RDHonest data documentation](https://search.r-project.org/CRAN/refmans/RDHonest/html/headst.html).
Its pinned source, transformation, observations, cutoff and question are unchanged.

## Interpretation constraints

- Rock the Vote keeps one observation per randomized cable system. The stratum
  fixed-effects estimate weights strata according to within-stratum treatment
  variation; it is not automatically an equally weighted population-person effect.
- NHEFS is observational. Complete-case targeting does not remove confounding or
  selection bias, and the result is not advice about smoking cessation.
- Castle selection uses the adoption schedule, not homicide outcomes. Both groups
  omit 2006. Periods 0–9 represent retained years 2000–2005 and 2007–2010; event
  offsets are retained waves, not elapsed years across the gap. Parallel trends
  remains an assumption.
- Head Start estimates eligibility for grant-writing assistance, not funding or
  program participation. Missing running values and outcomes must be reflected
  in the preparation and estimation receipts.

The fresh acceptance output is recorded locally at
`output/four-verified-20260909/README.md`. The earlier output at
`output/four-new-20260909/README.md` is historical evidence and does not count as
four accepted reports. These generated run directories are not included in Git;
a fresh clone does not contain those local evidence files.
