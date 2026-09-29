# families

One package per family, and the registry that lists them. The core packages beneath never import a family; they take the
families as an argument or reach them through `registry.REGISTRY`.

- `registry.py`: every family the desk knows, in the order the routing lists them. `needs()` for the matrix, `knowledge()` for the
  routing, `lanes()` for the compiled lanes.
- `base.py`: `FamilyDef` (the one object a package contributes: its knowledge, its needs, its design class and block builder, its
  probes, its lane), `Family` (the yaml prose the routing judges the data against), `load_family_yaml`, and the stub lane a declared
  family answers with.
- `adjustment/`, `diff_in_diff/`, `discontinuity/`: the built families, each with its own README.
- `declared/`: instrument, interrupted_series, synthetic_control, root_cause: a yaml each, so the routing can rule them in or out,
  and a probe where one applies. No lane.
- `tests/`: `test_core_names_no_family.py` greps the core for any family or engine name.

Adding a family is one package (`family.yaml`, `design.py`, `handoff.py`, `probes.py`, `postviz.py`, `lane/`, `evals/`, `tests/`) and
one line in the registry. See [docs/architecture.md](../../docs/architecture.md) and [ADR 0005](../../docs/adr/0005-one-folder-per-family.md).
