# Current workflow diagrams

The trajectory SVGs are generated from each lane's `rungs.yaml` by `make docfigs`. Amber nodes call a model; blue nodes run code. Each row shows the allowed exits, including an explicit stop where permitted.

- [Adjustment](adjustment-trajectory.svg)
- [Difference in differences](diff_in_diff-trajectory.svg)
- [Discontinuity](discontinuity-trajectory.svg)

Endpoint requirements, proof obligations and terminal conditions are generated separately by `make ladder` under `docs/ladders/`.
