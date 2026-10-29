// Row shaping for the inspector's tables. Pure: takes the session's views, returns plain rows.
import { cell, num } from "../fmt";
import type { CheckView, RunView } from "../types";

export interface EstimateRow {
  design: string;
  endpoint: string;
  contrast: string;
  method: string;
  value: string;
  ci: string;
  n: number | null;
  n_treated: number | null;
  n_control: number | null;
  p: string;
  primary: boolean;
  secondary: boolean;
  error: string | null;
}

export function primaryEstimate(run: RunView) {
  return run.estimates.find((e) => e.design_id === run.lead_design_id && !e.secondary && !e.modifier) ?? null;
}

/** The primary estimate first, then the rest in the order the lane wrote them. */
export function estimateRows(run: RunView): EstimateRow[] {
  const primary = primaryEstimate(run);
  const rows = run.estimates.map((e) => ({
    design: e.design_id,
    endpoint: e.endpoint_id,
    contrast: e.contrast ?? "—",
    method: e.method ?? "—",
    value: num(e.value),
    ci: e.confidence_set
      ? `${e.confidence_set.status}: ${e.confidence_set.regions.map(([lo, hi]) => `[${num(lo)}, ${num(hi)}]`).join(" ∪ ") || e.confidence_set.detail}`
      : `[${num(e.ci_low)}, ${num(e.ci_high)}]`,
    n: e.n,
    n_treated: e.n_treated,
    n_control: e.n_control,
    p: e.p_value == null ? "—" : `${num(e.p_value)}${e.p_value_source ? ` (${e.p_value_source})` : ""}`,
    primary: e === primary,
    secondary: e.secondary,
    error: e.error,
  }));
  return [...rows.filter((r) => r.primary), ...rows.filter((r) => !r.primary)];
}

export interface CheckRow {
  contrast: string;
  name: string;
  level: string;
  value: string;
  threshold: string;
  detail: string;
}

export function checkRows(checks: CheckView[]): CheckRow[] {
  return checks.map((c) => ({
    contrast: c.contrast ?? "—",
    name: c.name,
    level: c.level ?? "—",
    value: cell(c.value),
    threshold: cell(c.threshold),
    detail: c.detail ?? "",
  }));
}

export interface FalsificationRow {
  contrast: string;
  refuter: string;
  kind: string;
  passed: "passed" | "failed" | "n/a";
  p_value: string;
  new_effect: string;
  detail: string;
}

export function falsificationRows(run: RunView): FalsificationRow[] {
  return run.falsifications.map((x) => ({
    contrast: x.contrast ?? "—",
    refuter: x.refuter,
    kind: x.design_id,
    passed: x.passed === null ? "n/a" : x.passed ? "passed" : "failed",
    p_value: num(x.p_value),
    new_effect: num(x.new_effect),
    detail: x.detail ?? "",
  }));
}

export function decisionWhy(run: RunView): string | null {
  const w = run.decision.why;
  return w === null || w === undefined || w === "" ? null : String(w);
}

export function decisionOver(run: RunView): { family: string; reason: string }[] {
  const over = run.decision.over;
  if (!over || typeof over !== "object") return [];
  return Object.entries(over as Record<string, unknown>).map(([family, reason]) => ({ family, reason: String(reason) }));
}

export interface DeclineRow {
  address: string;
  about: string;
  kind: string;
  packValue: string;
  took: string;
  reason: string;
  check: string;
}

// Where the lane did not take the pack as given, one row each, in the order the lane recorded them.
export function declineRows(run: RunView): DeclineRow[] {
  return (run.declines ?? []).map((d) => ({
    address: d.address,
    about: d.about,
    kind: d.kind,
    packValue: d.pack_value ?? "—",
    took: d.took ?? "—",
    reason: d.reason,
    check: d.check,
  }));
}
