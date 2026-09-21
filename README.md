# Causal desk

One CSV and one question of the form "what did this change do to that outcome". The desk asks the question first,
validates it against the file, asks one thing per turn until a family of analysis stands, runs that family's lane on a
context pack, and talks about the result in the person's own words.

Three families are built: adjustment (DoWhy), diff-in-diff (pyfixest), discontinuity (rdrobust). Four more are
declared so the routing can rule them in or out.

## Run it

```bash
make dev-api      # the API on :8000
make dev-web      # the page on :5173, proxying /api
```

Or one run from the command line: `uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"`.

Copy `.env.example` to `.env`; only LangSmith needs a key, and only for tracing.

## Check it

```bash
make check        # everything CI runs: lint, types, tests, the web, the schema
```

`make` alone lists the targets.

## Read about it

- [docs/architecture.md](docs/architecture.md): the layers, a family package, the stores, the wire, the checks.
- [docs/adr/](docs/adr/): the decisions the layout rests on.
- [causal_agent/README.md](causal_agent/README.md): the packages; each has its own README.
