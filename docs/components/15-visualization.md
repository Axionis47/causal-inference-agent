# 15. Visualization and result presentation

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md) · [Evidence and discrepancies](evidence.md)

## Purpose and boundary

Connects recorded analytical evidence to figures, pictures, tables and cited explanations. There is no single visualization implementation: deterministic analysis figures, model-written exploration scripts and frontend rendering have separate responsibilities.

## Internal parts

| Path | Producers | Consumers |
|---|---|---|
| Structured figures | [FigureSpec](../../causal_agent/viz/spec.py), [common postviz](../../causal_agent/viz/postviz/common.py), family artifacts/postviz | [figure gate](../../causal_agent/lane/figures.py), desk material, [SVG renderer](../../web/src/components/Figure.tsx) |
| Requested pictures/tables | Exploration scripts and artifact client | [ArtifactCard](../../web/src/components/ArtifactCard.tsx), artifact API, chat material |
| Run inspection | [run projections](../../causal_agent/server/views.py) | [AnalysisCard](../../web/src/components/AnalysisCard.tsx), [ArtifactCard](../../web/src/components/ArtifactCard.tsx) |
| Cited discussion | [material](../../causal_agent/desk/material.py), [pull](../../causal_agent/desk/pull.py), explainer gates | [CitedText](../../web/src/components/CitedText.tsx) and transcript |

## Inputs and outputs

`FigureSpec` carries ID, kind, title, axis labels, series, reference marks, graph nodes/edges, note and source addresses. Kinds are bars, lines, points, density, interval and graph. An exploration `Artifact` instead carries numeric facts and file references such as `figure.png`. Run projections retain estimates, inference, checks, sensitivities and limitations.

## Contracts

| Boundary | Enforcement | Failure |
|---|---|---|
| Family figure → stored figure | `check_spec` checks graph references or nonempty values and resolvable `draws_on` addresses | Invalid figure dropped with a `Decline` |
| Figure → UI | Typed `FigureSpec`; server-computed series keys preserve addresses | Rendering helpers handle missing values and supported kinds |
| Artifact → image/table | Artifact file list and API URL projection | Missing/disallowed file yields 404 |
| Material → prose | Cites must resolve; numeric statements checked against supplied addressed facts | Re-prompt or fallback rather than invented evidence |

Figure validation checks structure and cited sources; it is not a general statistical proof of every plotted value. Family producers are responsible for deriving values from their recorded results.

## Decision authority

Analysis figure construction and SVG rendering are code. Exploration picture generation involves model-written scripts and a gated summary. Result interpretation and chat involve model judgments constrained by cited records. UI formatting never selects a replacement lead or changes the estimand.

## Dependencies

Shared viz sits below lane/families; family-specific figures remain inside their family. The server projects existing results. React displays structured specs or artifact files. This component relies on [artifact ownership](09-persistence.md) and [exploration](14-exploration.md), rather than becoming another storage or estimation layer.

## State and storage

Structured specs are stored in run/design `figures.json` and cited by `figure:` addresses. Requested pictures are files on the artifact shelf cited by `artifact:` addresses. Chat material binds explanations to the saved run context rather than allowing later live-memory edits to rewrite the old result.

## Worked walkthrough

A completed adjustment run produces a graph and comparable-estimate figures → the gate validates addresses → the desk includes them in the record → the server projects them → React renders SVG. A later request for a custom distribution plot uses exploration and produces an artifact picture instead of adding a new estimator or changing the recorded figure contract.

## Verification

**Code-verified:** FigureSpec, figure gate, artifact projection and render paths. **Test-supported:** [spec](../../causal_agent/viz/tests/test_spec.py), [postviz](../../causal_agent/viz/tests/test_postviz.py), [figure gates](../../causal_agent/lane/tests/test_figures.py), [renderer helpers](../../web/src/components/Figure.test.ts), [figure helpers](../../web/src/figure.test.ts), [material](../../causal_agent/desk/tests/test_material.py). **Browser coverage:** the consolidated Compose journey exercises context review and chat inspectors. Artifact numerical grounding and file access are verified separately; this does not establish appearance across devices.
