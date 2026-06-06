// Canonical specialist roster for the terminal JobPage left rail.
// Order matches the analysis pipeline. See backend/context/agents.md §2.

export interface AgentRow {
  /** Stable key matching the backend agent registration name. */
  key: string;
  /** Short label rendered in the roster row (mono, lowercase). */
  label: string;
  /** One-line "answers" string from AgentCapability; used as the empty headline. */
  answers: string;
}

export const SPECIALIST_ROSTER: ReadonlyArray<AgentRow> = [
  { key: 'dataset_inspector',    label: 'dataset_inspector',    answers: 'which file in the bundle' },
  { key: 'data_profiler',        label: 'data_profiler',        answers: 'what shape is this data' },
  { key: 'domain_knowledge',     label: 'domain_knowledge',     answers: 'what does the literature say' },
  { key: 'eda',                  label: 'eda',                  answers: 'distributional shape' },
  { key: 'data_repair',          label: 'data_repair',          answers: 'salvage missing values' },
  { key: 'causal_discovery',     label: 'causal_discovery',     answers: 'what causal structure' },
  { key: 'dag_expert',           label: 'dag_expert',           answers: 'refine + adjustment set' },
  { key: 'confounder_discovery', label: 'confounder_discovery', answers: 'which confounders matter' },
  { key: 'ps_diagnostics',       label: 'ps_diagnostics',       answers: 'is overlap good enough' },
  { key: 'effect_estimator',     label: 'effect_estimator',     answers: 'treatment effect, many methods' },
  { key: 'sensitivity_analyst',  label: 'sensitivity_analyst',  answers: 'how stable under hidden confounding' },
  { key: 'critique',             label: 'critique',             answers: 'accept or loop back' },
  { key: 'notebook',             label: 'notebook',             answers: 'render final notebook' },
];

export type AgentTone = 'pending' | 'live' | 'ok' | 'failed';

export interface AgentStatusMap {
  [agentKey: string]: AgentTone;
}

/**
 * Derive each agent's display tone from the live job state.
 *
 * Rules (per aesthetics.md §2 — the live/settled axis):
 *   - currentAgent       -> live      (amber pulse), or failed if the job failed there
 *   - completed event    -> ok        (mint dot)
 *   - everything else    -> pending   (edge-subtle dot)
 *
 * `completedAgents` is a Set of agent_name strings that emitted an
 * `agent_completed` event.
 */
export function deriveAgentStatuses(
  currentAgent: string | null | undefined,
  completedAgents: Set<string>,
  jobFailed: boolean,
): AgentStatusMap {
  const out: AgentStatusMap = {};
  for (const row of SPECIALIST_ROSTER) {
    if (currentAgent === row.key) {
      out[row.key] = jobFailed ? 'failed' : 'live';
    } else if (completedAgents.has(row.key)) {
      out[row.key] = 'ok';
    } else {
      out[row.key] = 'pending';
    }
  }
  return out;
}

/** Returns the headline string to show next to an agent row, given its latest event. */
export function describeEventForRow(
  latestEvent: { event_type?: string; data?: Record<string, unknown> } | undefined,
  fallback: string,
): string {
  if (!latestEvent) return fallback;
  const data = latestEvent.data ?? {};
  if (typeof data.headline === 'string') return data.headline;
  if (typeof data.status === 'string') return `event: ${data.status}`;
  return latestEvent.event_type ?? fallback;
}
