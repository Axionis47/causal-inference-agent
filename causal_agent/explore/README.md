# explore

The chatbot supplies a question, the entire selected CSV version, context and optional published artifact references. The trusted client resolves these inputs and launches the A2A agent. The agent cannot select another dataset or search unrelated storage.

- `contracts.py`: whole-data `ExploreRequest`, separate `TransformRequest`, typed `Exploration` and complete execution `Trail`.
- `inputs.py`: exact-byte checksum validation, task-local copies and the supplied-resource manifest.
- `inspection.py`: bounded input/output inspection, image pixels for the model, and generated Python execution.
- `sandbox.py`: Docker-only generated-code execution. Read-only inputs/root, no network, non-root user, bounded resources, cancellation and output validation. No host Python fallback.
- `agent.py`, `prompts.py`, `checks.yaml`: shared episode, eight scripts, sixteen inspections, three final-record attempts, sixty seconds per script.
- `gates.py`: successful-step evidence, numerical grounding, parsed products and inspected plots. Missing data returns explicit needs to the chatbot.
- `client.py`, `server.py`: A2A transport, lifecycle and the owned publication adapter.
- `project.py`: row identities and withheld outcomes for the separate transformation workflow. User-facing exploration does not use it.
- `diagnostics.py`: shared diagnostic execution; each family retains responsibility for restricting its supplied data and its three-call budget.

Build once with `make explore-image`. Docker Desktop (Linux containers) or Docker Engine must be running. `EXPLORE_IMAGE` selects the installed image. The image pins Python and numerical libraries and installs pandas, NumPy, SciPy, statsmodels, matplotlib, scikit-learn and Pillow. Windows users can use the equivalent `docker build -t causal-desk-explore -f docker/explore.Dockerfile .` command without Make.

The integration service publishes explanations and registered files under their owning conversation in PostgreSQL and private S3. The artifact interface has no filesystem catalogue fallback. Compose keeps the A2A agent and restricted Docker worker available at runtime. See [exploration](../../docs/exploration.md) and [container deployment](../../docs/production/containerization.md).
