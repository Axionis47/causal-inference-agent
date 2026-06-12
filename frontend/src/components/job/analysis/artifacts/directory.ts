// The artifact directory: what every artifact the spine emits is, and which
// inline view renders it. Resolution: exact id -> prefix family -> kind
// fallback, so unknown artifacts still get a presentable default view.
// Mirrors the backend inventory (analysis_v2 agents); when an agent gains an
// artifact, add its id here.

import type { AnalysisArtifactKind, AnalysisArtifactView } from '../../../../services/api';

export type ArtifactRenderer = 'json' | 'table' | 'markdown' | 'plot' | 'link';

export interface DirectoryEntry {
  /** One line on what this artifact is; shown in the expanded view header. */
  description: string;
  /** Override the kind-derived renderer (bespoke views hook in here). */
  renderer?: ArtifactRenderer;
}

const EXACT: Record<string, DirectoryEntry> = {
  // s1 intake
  'intake/causal_spec': { description: 'The parsed question as a typed spec: type, treatment, outcome, covariates.' },
  'intake/question_type_candidates': { description: 'Question types considered, with the chosen one and its confidence.' },
  'intake/missing_info': { description: 'What the question left unstated.' },
  'intake/summary': { description: 'The intake reasoning summary.' },
  // s2 profiling
  'profiling/dataset_profile': { description: 'Deterministic dataset profile: rows, columns, semantic types, warnings.' },
  'profiling/column_inventory': { description: 'Per-column inventory: type, missingness, distinct counts.' },
  'profiling/preview_table': { description: 'The first rows of the dataset as loaded.' },
  'profiling/missingness': { description: 'Missing share by column.' },
  'profiling/numeric_distributions': { description: 'Histograms of the numeric columns.' },
  'profiling/summary': { description: 'Narrative profile summary.' },
  // s3 design detection
  'design/candidates': { description: 'Candidate designs with confidence and rationale.' },
  'design/tool_eligibility': { description: 'Which method lanes are enabled, conditional, or disabled, and why.' },
  'design/disabled_methods': { description: 'Disabled lanes and the requirement each one is missing.' },
  'design/required_missing_fields': { description: 'Fields a conditional lane still needs before it can run.' },
  'design/refined_spec': { description: 'The causal spec after design detection refined it.' },
  // s4 targeted eda
  'eda/plan': { description: 'Which EDA checks ran for the target design, and why.' },
  'eda/summary': { description: 'All EDA check results in one typed summary.' },
  'eda/warnings': { description: 'Warnings the EDA checks raised.' },
  'eda/story': { description: 'The EDA narrative: data patterns and design-relevant signals.' },
  // s5 plan critic
  'plan/critique': { description: 'The plan gate verdict with its reasons.' },
  'plan/confirmation_card': { description: 'The card shown when the plan needed user confirmation.' },
  'plan/method_plan': { description: 'The locked estimation plan: lane, estimator, variables.' },
  // s7 method lane
  'method/estimate_result': { description: 'The primary estimate with effects and standard errors.' },
  // s8 diagnostics
  'diagnostics/report': { description: 'Stress-test results on the fitted model.' },
  'diagnostics/sensitivity': { description: 'Sensitivity checks and the robustness verdict.' },
  // s9 claim critic
  'claims/critique': { description: 'Claim strength and the language guardrails behind it.' },
  'claims/allowed_language': { description: 'Phrases the report may and may not use.' },
  'claims/limitations': { description: 'Limitations the report must carry.' },
  // s10 report + notebook build
  'report/final_report': { description: 'The final narrative report.' },
  'report/final_report_json': { description: 'The report data: question, claim strength, estimate, limitations.' },
  'report/dashboard_payload': { description: 'Data behind the results dashboard.' },
  'report/artifacts_index': { description: 'Index of everything this run emitted.' },
  'notebook/config': { description: 'Reproducibility config the notebook runs with.' },
  'notebook/causal_analysis': { description: 'The 13-section analysis notebook.' },
  // s11 notebook verify
  'notebook/verified': { description: 'The executed notebook with cell outputs.' },
  'notebook/html_preview': { description: 'Browser preview of the executed notebook.' },
  'notebook/execution_log': { description: 'Notebook execution attempts and repairs.' },
};

// Dynamic families: per-check EDA outputs and per-lane method outputs carry
// generated names, so they resolve by prefix.
const PREFIX: Array<[string, DirectoryEntry]> = [
  ['eda/', { description: 'A targeted EDA check output.' }],
  ['method/', { description: 'Output from the estimation lane.' }],
];

const KIND_RENDERER: Record<AnalysisArtifactKind, ArtifactRenderer> = {
  json: 'json',
  table: 'table',
  markdown: 'markdown',
  plot: 'plot',
  notebook: 'link',
  html: 'link',
};

export interface ResolvedArtifact {
  description: string | null;
  renderer: ArtifactRenderer;
}

export function resolveArtifact(artifact: AnalysisArtifactView): ResolvedArtifact {
  const entry =
    EXACT[artifact.artifact_id] ??
    PREFIX.find(([p]) => artifact.artifact_id.startsWith(p))?.[1] ??
    null;
  return {
    description: entry?.description ?? null,
    renderer: entry?.renderer ?? KIND_RENDERER[artifact.kind],
  };
}
