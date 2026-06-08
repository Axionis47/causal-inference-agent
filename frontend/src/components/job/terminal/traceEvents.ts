// Hydrate the terminal panes from stored agent traces when there is no live SSE
// stream. A running job feeds the tape/rail from SSE events; a completed or
// reloaded job has no live events, so without this it sits on "awaiting first
// event" forever. We map its persisted traces into the same AgentEvent shape the
// panes already consume, so the whole panel comes alive from history.

import type { AgentEvent, AgentTrace } from '../../../services/api';

/** Map one persisted ReAct trace to the AgentEvent shape the tape renders.
 *  The ReAct loop prefixes step actions with "step_N_"; strip it so a row reads
 *  "classify_variable_role", not "step_19_classify_variable_role". */
export function traceToEvent(trace: AgentTrace): AgentEvent {
  const action = (trace.action || 'step').replace(/^step_\d+_/, '');
  return {
    timestamp: trace.timestamp,
    agent_name: trace.agent_name,
    event_type: action,
    data: {
      headline: action,
      reasoning: trace.reasoning,
      tools_called: trace.tools_called,
      duration_ms: trace.duration_ms,
    },
  };
}

function bumpTimestamp(ts: string): string {
  const ms = new Date(ts).getTime();
  return Number.isNaN(ms) ? ts : new Date(ms + 1).toISOString();
}

/** Build a chronological AgentEvent list from persisted traces, plus a synthetic
 *  agent_completed marker per agent (just after that agent's last trace) so a
 *  reloaded finished job shows completed tones in the rail rather than all-idle. */
export function tracesToEvents(traces: AgentTrace[]): AgentEvent[] {
  if (traces.length === 0) return [];

  const events: AgentEvent[] = traces.map(traceToEvent);

  // Traces are chronological, so the last write per agent is its final step.
  const lastTsByAgent = new Map<string, string>();
  for (const t of traces) {
    if (t.agent_name) lastTsByAgent.set(t.agent_name, t.timestamp);
  }
  for (const [agent, ts] of lastTsByAgent) {
    events.push({
      timestamp: bumpTimestamp(ts),
      agent_name: agent,
      event_type: 'agent_completed',
      data: { success: true },
    });
  }

  return events.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}
