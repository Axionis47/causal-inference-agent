# chat — the desk

One conversation before, through, and after the analysis. Before the run it is the interview (`intake/interview`), which settles the claims and raises the ready flag. `run` hands the dataset and the question to the router and the lanes as a black box. After the run the person can ask what was found, why this design, what a flag or a falsification means, what would change the answer; change a claim about the data, which re-enters the interview and reruns; ask a new question of the same data; or say done.

```
START ─ interview ─ run ─ brief ─ talk ─(interrupt)─ turn ─┬─ answer ──────────────── talk
                                                          ├─ revise ─ interview ─ run ─ brief ─ talk
                                                          ├─ requestion ─ run ─ brief ─ talk
                                                          └─ done ─ END
```

## Facts, judgements, gates

- **Facts**: `run` (`pipeline.py`, the router on a fresh thread, a `RunRecord` out), `brief` (`material.py`, the opening message rendered from artifacts by template, every value with its address, then-and-now when a previous run exists), `revise` (applies the claim updates through the interview's own `_apply` with the person's cite and re-enters the interview, which re-checks and re-probes), `requestion`.
- **Judgement**: `turn`, one call, an `AfterReply` with a kind (`answer`, `revise`, `requestion`, `done`), the text, its cites, every number it states with the address it comes from, the claim updates a revision implies, or the new question.
- **Gate**: cites resolve in the material; every number attached to an address matches the artifact within 1 percent; every number in the text is either attached or present in the material, so nothing is stated the run did not produce; a revise carries at least one legal claim update; a requestion carries the question. Three tries, then a reply that says what could not be grounded.
- **What the person cannot do**: change an estimator, a bandwidth, or a covariate set directly. The reply says which claim would change it, because designs follow from claims.

## Material

`material.render(run, claims, previous)` reads the run record, not the lane state, so it works for stored runs. Addresses match the lanes' interpret addresses (`design.*`, `check:<c>.<name>`, `estimate:<c>[.<method>].value|ci|n`, `placebo:` or `refute:`), plus `interpretation:<c>.answer|caveat:<i>`, `decision.family|assumption|why|over:<family>`, `feasibility.*`, `primary.p|h|b` for the discontinuity lane, `claim:*`, and `run:<i>.effect` for the previous run. `numbers` keys every numeric value by address for the gate.

## Run it

```bash
uv run python -m causal_agent.chat data/raw/senate-incumbency/senate.csv --name senate3 --context notes.md --question "Does winning narrowly raise the next vote share?"
uv run langgraph dev   # graph "chat" in Studio
uv run pytest causal_agent/chat -q
uv run python -m causal_agent.server   # the web page over the same graph, see causal_agent/server/README.md
```

## Known limits

- The analysis runs in a subprocess (`python -m causal_agent.router.run … --json-file`). In-process, the discontinuity lane ended the parent process silently on macOS once the profiler had run in it; the subprocess is also what makes the pipeline a black box to the chat.
- The interview subgraph's state is visible on the parent thread only through `get_state(cfg, subgraphs=True)` while it is interrupted; the CLI reads the interrupt payload and needs nothing else.
- A revision reruns the whole pipeline. Nothing is cached between runs.
- No figures yet; the figure-note judgement is a later step.
