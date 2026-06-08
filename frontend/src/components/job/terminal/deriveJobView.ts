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
  /** Latest finding per agent. The rail prefers its headline over the generic
   *  latest event, so a finished agent shows "quality 82/100" not "eda done"
   *  (agent_completed arrives after agent_finding and would otherwise win). */
  findingByAgent: Map<string, AgentEvent>;
  /** Events in newest-first order for the tape pane. */
  reverseEvents: AgentEvent[];
  /** The agent the focus pane shows: the selected one, else the dispatched one. */
  focusAgent: string | null;
  /** Latest event for the focus agent, or undefined if none yet. */
  focusLatest: AgentEvent | undefined;
  /** The focus agent's latest finding event (brief headline + flags), if any. */
  focusFinding: AgentEvent | undefined;
  /** The focus agent's latest challenge event (refusal/flag + issues), if any. */
  focusChallenge: AgentEvent | undefined;
  /** Agents that raised at least one challenge, for the rail indicator. */
  challengedAgents: Set<string>;
  /** Convenience: did the job fail? */
  failed: boolean;
}

export function deriveJobView(
  job: JobDetail,
  agentEvents: AgentEvent[],
  nowMs: number,
  selectedAgent: string | null = null,
): JobView {
  const failed = job.status === 'failed';
  const completedAgents = new Set(
    agentEvents.filter(e => e.event_type === 'agent_completed').map(e => e.agent_name ?? ''),
  );
  const agentTones = deriveAgentStatuses(job.current_agent, completedAgents, failed);

  const latestByAgent = new Map<string, AgentEvent>();
  const findingByAgent = new Map<string, AgentEvent>();
  const challengeByAgent = new Map<string, AgentEvent>();
  for (const ev of agentEvents) {
    if (!ev.agent_name) continue;
    latestByAgent.set(ev.agent_name, ev);
    if (ev.event_type === 'agent_finding') findingByAgent.set(ev.agent_name, ev);
    else if (ev.event_type === 'agent_challenge') challengeByAgent.set(ev.agent_name, ev);
  }

  // Focus the selected agent (clickable rail) when one is set, otherwise the
  // currently dispatched agent. A selection survives the run finishing, so a
  // finished agent's findings stay inspectable.
  const focusAgent = selectedAgent ?? job.current_agent ?? null;
  const focusLatest = focusAgent ? latestByAgent.get(focusAgent) : undefined;
  const focusFinding = focusAgent ? findingByAgent.get(focusAgent) : undefined;
  const focusChallenge = focusAgent ? challengeByAgent.get(focusAgent) : undefined;

  return {
    elapsed: formatElapsed(job.created_at, nowMs),
    agentTones,
    latestByAgent,
    findingByAgent,
    reverseEvents: [...agentEvents].reverse(),
    focusAgent,
    focusLatest,
    focusFinding,
    focusChallenge,
    challengedAgents: new Set(challengeByAgent.keys()),
    failed,
  };
}
