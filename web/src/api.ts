/** The six endpoints, typed. Nothing else talks to the network. */

const BASE = "/api";

export type Column = {
  name: string;
  dtype: string;
  missing: number;
  n_unique: number;
  numeric: boolean;
  binary: boolean;
  datelike: boolean;
  low: number | null;
  high: number | null;
};

export type Option = {
  lane: string;
  available: boolean;
  reason: string;
  assumption: string;
  needs: Record<string, string>;
};

export type Intake = {
  outcome: string;
  treatment: string;
  group: string;
  period: string;
  time_column: string;
  running_variable: string;
  cutoff: number | null;
  question_family: string;
  confidence: string;
  reasoning: string;
  exposure: string;
};

export type Role = { column: string; role: string; why: string };

export type Recommendation = {
  lane: string;
  step: number;
  reasoning: string;
  assumption: string;
  confidence: string;
  missing: string;
  runner_up: string;
  failed: string;
};

export type Job = {
  id: string;
  status: "running" | "waiting_for_you" | "completed" | "failed";
  question: string | null;
  n_rows: number | null;
  columns: Column[];
  intake: Intake | null;
  menu: Option[];
  roles: Record<string, Role>;
  recommendation: Recommendation | null;
  suggestions: Record<string, Record<string, unknown>>;
  source: string | null;
  source_note: string | null;
  error: string | null;
};

export type Estimate = {
  estimand: string;
  value: number;
  n: number;
  estimator: string;
  se: number | null;
  ci_low: number | null;
  ci_high: number | null;
  p_value: number | null;
  notes: string[];
};

export type Result = {
  id: string;
  status: string;
  lane: string | null;
  estimate: Estimate | null;
  strength: string | null;
  headline: string | null;
  narrative: string | null;
  error: string | null;
};

export type Dataset = { name: string; columns: string[]; n_columns: number };

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} on ${path}`);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} on ${path}`);
  return r.json();
}

export const api = {
  datasets: () => get<Dataset[]>("/datasets"),
  createJob: (body: {
    dataset?: string;
    kaggle?: string;
    question: string;
    context: string;
  }) => post<{ id: string; note: string }>("/jobs", body),
  job: (id: string) => get<Job>(`/jobs/${id}`),
  chooseDesign: (id: string, lane: string, kwargs: Record<string, unknown>) =>
    post<{ id: string }>(`/jobs/${id}/design`, { lane, kwargs }),
  result: (id: string) => get<Result>(`/jobs/${id}/result`),
  notebookUrl: (id: string) => `${BASE}/jobs/${id}/notebook`,
};

/** The five event names the tape depends on. Pinned server-side by
 *  test_events.py: renaming one keeps every status at 200 and silently stops
 *  this from updating. */
export type TapeEvent = {
  event: "stage_started" | "stage_done" | "waiting_for_you" | "completed" | "failed";
  stage?: string;
  detail?: string;
  reason?: string;
};

export function openTape(id: string, onEvent: (e: TapeEvent) => void): () => void {
  const src = new EventSource(`${BASE}/jobs/${id}/stream`);
  const names: TapeEvent["event"][] = [
    "stage_started",
    "stage_done",
    "waiting_for_you",
    "completed",
    "failed",
  ];
  for (const name of names) {
    src.addEventListener(name, (ev) => {
      let payload: Record<string, unknown> = {};
      try {
        payload = JSON.parse((ev as MessageEvent).data || "{}");
      } catch {
        /* a heartbeat or an empty frame; the event name is the signal */
      }
      onEvent({ ...payload, event: name } as TapeEvent);
    });
  }
  return () => src.close();
}
