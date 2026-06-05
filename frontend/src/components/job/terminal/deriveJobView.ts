// Derives every value the terminal panes need from raw job + event data.
// Pure: no hooks, no I/O. Named without `use` prefix because JobPage calls
// it after early returns; React's lint would otherwise flag it.

import type { AgentEvent, JobDetail } from '../../../services/api';
import { deriveAgentStatuses, type AgentStatusMap } from './agents';
import { formatElapsed } from './format';

export interface JobView {
  /** HH:MM:SS since job.created_at, ticking. */
  elapsed: string;
  /** Per-agent tone map for the phase strip + agents rail + focus pane. */
  agentTones: AgentStatusMap;
  /** Latest event seen per agent_name, for rail subtitles + focus "last event". */
  latestByAgent: Map<string, AgentEvent>;
  /** Events in newest-first order for the tape pane. */
  reverseEvents: AgentEvent[];
  /** The currently dispatched agent, or null when idle/terminal. */
  focusAgent: string | null;
  /** Latest event for the focus agent, or undefined if none yet. */
  focusLatest: AgentEvent | undefined;
  /** Convenience: did the job fail? */
  failed: boolean;
}

export function deriveJobView(
  job: JobDetail,
  agentEvents: AgentEvent[],
  nowMs: number,
): JobView {
  const failed = job.status === 'failed';
  const completedAgents = new Set(
    agentEvents.filter(e => e.event_type === 'agent_completed').map(e => e.agent_name ?? ''),
  );
  const agentTones = deriveAgentStatuses(job.current_agent, completedAgents, failed);

  const latestByAgent = new Map<string, AgentEvent>();
  for (const ev of agentEvents) {
    if (ev.agent_name) latestByAgent.set(ev.agent_name, ev);
  }

  const focusAgent = job.current_agent ?? null;
  const focusLatest = focusAgent ? latestByAgent.get(focusAgent) : undefined;

  return {
    elapsed: formatElapsed(job.created_at, nowMs),
    agentTones,
    latestByAgent,
    reverseEvents: [...agentEvents].reverse(),
    focusAgent,
    focusLatest,
    failed,
  };
}
