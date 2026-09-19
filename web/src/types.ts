import type { FigureSpec } from "./figure";

// Mirrors of causal_agent/server/models.py.

export interface NumericShape {
  min: number;
  p25: number;
  p50: number;
  p75: number;
  max: number;
  mean: number;
}

export interface TopValue {
  value: string;
  count: number;
  share: number;
}

export interface DatetimeShape {
  first: string;
  last: string;
  frequency: string | null;
}

export interface Sentinel {
  value: string;
  count: number;
  reason: string;
}

export interface ColumnSummary {
  name: string;
  key: string;
  kind: string;
  nulls: number;
  null_rate: number;
  distinct: number;
  constant: boolean;
  examples: string[];
  numeric: NumericShape | null;
  top_values: TopValue[];
  datetime: DatetimeShape | null;
  sentinels: Sentinel[];
  issues: string[];
}

export interface ProfileOut {
  upload_id: string;
  filename: string;
  rows: number;
  columns: ColumnSummary[];
  head: string[][];
  duplicate_rows: number;
  candidate_keys: string[][];
  grain: string[] | null;
  co_missing: string[][];
  issues: string[];
}

export interface SessionBrief {
  stage: string;
  phase: string;
  runs: number;
}

export interface DatasetSummary {
  name: string;
  title: string;
  csv: string;
  rows: number | null;
  columns: number | null;
  created_at: string | null;
  shipped: boolean;
  has_claims: boolean;
  question: string | null;
  session: SessionBrief | null;
}

export interface DatasetCreate {
  name: string;
  title: string;
  upload_id: string;
}

export interface QuestionView {
  keys: string[];
  field: string | null;
  kind: "confirm" | "choose" | "open" | "columns" | string;
  text: string;
  options: string[];
  evidence_cites: string[];
  because: string[];
}

export interface ClaimView {
  key: string;
  kind: string;
  status: "empty" | "drafted" | "confirmed" | "refuted" | "unknown" | "contradiction" | string;
  fields: Record<string, unknown>;
  source: string | null;
  evidence: string[];
  check_detail: string | null;
  asked: number;
  refutations: number;
}

export interface StatusView {
  table: Record<string, Record<string, string>>;
  surviving: string[];
  struck: Record<string, string>;
  required: string[];
  settled: string[];
  open: string[];
  ready: boolean;
  contradictions: string[];
}

export interface CheckView {
  contrast: string | null;
  name: string;
  level: string | null;
  value: unknown;
  threshold: unknown;
  detail: string | null;
}

export interface RefutationView {
  contrast: string | null;
  refuter: string;
  kind: string | null;
  passed: boolean | null;
  p_value: number | null;
  new_effect: number | null;
  detail: string | null;
}

export interface InterpretationView {
  contrast: string | null;
  answer: string;
  caveats: string[];
  cites: string[];
}

export interface EstimateView {
  contrast: string | null;
  method: string | null;
  value: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n: number | null;
  n_treated: number | null;
  n_control: number | null;
  secondary: boolean;
  error: string | null;
}

export interface RunView {
  index: number;
  question: string;
  family: string | null;
  specialist: string | null;
  status: string;
  run_id: string | null;
  effect: number | null;
  ci_low: number | null;
  ci_high: number | null;
  estimator: string | null;
  decision: Record<string, unknown>;
  decision_record: string;
  flags: CheckView[];
  checks: CheckView[];
  refutations: RefutationView[];
  interpretations: InterpretationView[];
  estimates: EstimateView[];
  feasibility: Record<string, unknown> | null;
  files: string[];
  what_if: Record<string, string>;
  differs: string[];
}

export interface Turn {
  role: "user" | "assistant" | "system";
  text: string;
  phase: string;
  at: string;
  kind: string | null;
  figure?: FigureSpec | null;
}

export interface Prompt {
  text: string;
  status: string;
  ready: boolean;
  open: string[];
  runs: number;
  phase: string;
  kind?: string | null;
}

export type Stage = "busy" | "waiting" | "ended" | "stale" | "error" | "new";

export interface SessionView {
  name: string;
  title: string;
  question: string | null;
  stage: Stage;
  phase: string;
  activity: { node: string; since: string } | null;
  ready: boolean;
  prompt: Prompt | null;
  questions: QuestionView[];
  claims: ClaimView[];
  status: StatusView | null;
  runs: RunView[];
  brief: string;
  transcript: Turn[];
  written: Record<string, unknown> | null;
  error: string | null;
}

export interface RunFiles {
  run_id: string;
  files: { name: string; size: number }[];
}
