# 5. One folder per family; the core names none

A family is one component. Its knowledge, the claims it needs, its design block and the code that fills it, its probes,
its pre-run and post-run figures, its lane, its evals and its tests live in `causal_agent/families/<name>/`, and the
package builds one `FamilyDef` the registry lists.

The core packages (`common`, `profile`, `memory`, `viz`, `lane`) never import a family. They take the families as an
argument (`memory.ops.probe`, `fit`, `open`) or offer a register a family fills at import (`common.contracts.DESIGNS`,
`viz.graph.FIGURES`). `import-linter` enforces both the layers and this rule, and a test greps the core for family and
engine names. Adding a family touches one package and one registry line.
