// One-line headline derivation for any AgentEvent. Used by tape rows and agents-rail subtitles.

import type { AgentEvent } from '../../../services/api';

/** Prefer data.headline when present; else synthesise a short string from event_type + agent_name. */
export function describeEvent(ev: AgentEvent): string {
  const t = ev.event_type;
  const d = ev.data ?? {};
  if (typeof d.headline === 'string') return d.headline;
  if (t === 'agent_started') return `${ev.agent_name ?? 'agent'} dispatched`;
  if (t === 'agent_completed') return `${ev.agent_name ?? 'agent'} done`;
  if (typeof d.status === 'string') return `${t} · ${d.status}`;
  return t;
}
