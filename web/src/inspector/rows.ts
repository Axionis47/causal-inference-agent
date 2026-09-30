// Row shaping for the inspector's tables. Pure: takes the session's views, returns plain rows.
import { cell, num } from "../fmt";
import type { CheckView, ClaimView, RunView, StatusView, StepView } from "../types";

export interface ClaimRow {
  key: string;
  kind: string;
  status: string;
  fields: { k: string; v: string }[];
  source: string | null;
  evidence: string[];
  check: string | null;
  asked: number;
  refutations: number;
}

export function claimRows(claims: ClaimView[]): ClaimRow[] {
  return claims.map((c) => ({
    key: c.key,
    kind: c.kind,
    status: c.status,
    fields: Object.entries(c.fields)
      .filter(([, v]) => v !== null && v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0))
      .map(([k, v]) => ({ k, v: Array.isArray(v) ? v.map(String).join(", ") : typeof v === "object" ? JSON.stringify(v) : String(v) })),
    source: c.source,
    evidence: c.evidence,
    check: c.check_detail,
    asked: c.asked,
    refutations: c.refutations,
  }));
}

export interface Matrix {
  families: { name: string; struck: string | null }[];
  rows: { kind: string; cells: Record<string, string> }[];
}

/** Kinds as rows in the order the claims introduce them, families as columns, one word per cell. */
export function statusMatrix(status: StatusView | null, claims: ClaimView[]): Matrix | null {
  if (!status) return null;
  const familyNames = Object.keys(status.table);
  if (!familyNames.length) return null;
  const kinds: string[] = [];
  const seen = new Set<string>();
  for (const c of claims)
    if (!seen.has(c.kind)) {
      seen.add(c.kind);
      kinds.push(c.kind);
    }
  for (const f of familyNames)
    for (const k of Object.keys(status.table[f]))
      if (!seen.has(k)) {
        seen.add(k);
        kinds.push(k);
      }
  return {
    families: familyNames.map((name) => ({ name, struck: status.struck[name] ?? null })),
    rows: kinds.map((kind) => ({ kind, cells: Object.fromEntries(familyNames.map((f) => [f, status.table[f][kind] ?? "not_needed"])) })),
  };
}

export interface EstimateRow {
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
  return run.estimates.find((e) => !e.secondary && !e.error) ?? null;
}

/** The primary estimate first, then the rest in the order the lane wrote them. */
export function estimateRows(run: RunView): EstimateRow[] {
  const primary = primaryEstimate(run);
  const rows = run.estimates.map((e) => ({
    contrast: e.contrast ?? "—",
    method: e.method ?? "—",
    value: num(e.value),
    ci: `[${num(e.ci_low)}, ${num(e.ci_high)}]`,
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

export interface RefutationRow {
  contrast: string;
  refuter: string;
  kind: string;
  passed: "passed" | "failed" | "n/a";
  p_value: string;
  new_effect: string;
  detail: string;
}

export function refutationRows(run: RunView): RefutationRow[] {
  return run.refutations.map((x) => ({
    contrast: x.contrast ?? "—",
    refuter: x.refuter,
    kind: x.kind ?? "—",
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

// ------------------------------------------------------------------ the journal, grouped by design run

export interface JournalRow {
  n: number;
  address: string;
  kind: string;
  by: string;
  at: string;
  note: string;
  run: number | null;
  hasFiles: boolean;
}
export interface JournalGroup {
  /** The design run the steps belong to; null for steps that led nowhere yet (no design written after them). */
  design: number | null;
  rows: JournalRow[];
}

/** Steps grouped by design run. A step before a design belongs to the design that follows it; a step after a run belongs to that
 * run, as the server marks it. Steps after the last run that lead to no design yet form a trailing group of their own. */
export function journalGroups(steps: StepView[]): JournalGroup[] {
  const groups: JournalGroup[] = [];
  let pending: JournalRow[] = [];
  const row = (s: StepView): JournalRow => ({
    n: s.n,
    address: s.address,
    kind: s.kind,
    by: s.by,
    at: s.at,
    note: s.note,
    run: s.run,
    hasFiles: !!s.run_id,
  });
  for (const s of steps) {
    if (s.design === null) {
      pending.push(row(s));
      continue;
    }
    let g = groups.find((x) => x.design === s.design);
    if (!g) {
      g = { design: s.design, rows: [] };
      groups.push(g);
    }
    if (pending.length) {
      g.rows.push(...pending);
      pending = [];
    }
    g.rows.push(row(s));
  }
  if (pending.length) groups.push({ design: null, rows: pending });
  return groups;
}
