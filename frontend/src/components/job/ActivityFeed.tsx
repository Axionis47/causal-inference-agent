import { useEffect, useRef } from 'react';
import { AgentEvent } from '../../services/api';

interface ActivityFeedProps {
  events: AgentEvent[];
}

const agentBadgeColors: Record<string, string> = {
  orchestrator: 'bg-purple-100 text-purple-700',
  data_profiler: 'bg-blue-100 text-blue-700',
  causal_discovery: 'bg-green-100 text-green-700',
  effect_estimator: 'bg-yellow-100 text-yellow-700',
  sensitivity_analyst: 'bg-orange-100 text-orange-700',
  critique: 'bg-red-100 text-red-700',
  notebook_generator: 'bg-indigo-100 text-indigo-700',
  dag_expert: 'bg-teal-100 text-teal-700',
  confounder_discovery: 'bg-cyan-100 text-cyan-700',
  ps_diagnostics: 'bg-lime-100 text-lime-700',
  data_repair: 'bg-rose-100 text-rose-700',
  domain_knowledge: 'bg-violet-100 text-violet-700',
  eda_agent: 'bg-emerald-100 text-emerald-700',
  effect_estimator_react: 'bg-amber-100 text-amber-700',
};

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour12: false });
  } catch {
    return '--:--:--';
  }
}

function actionText(event: AgentEvent): string {
  const name = event.agent_name.replace(/_/g, ' ');
  if (event.event_type === 'agent_started') return `${name} started`;
  if (event.event_type === 'agent_completed') return `${name} completed`;
  return `${name} — ${event.event_type}`;
}

export default function ActivityFeed({ events }: ActivityFeedProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to newest entry
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  // Keep only last 20 entries for display
  const visible = events.slice(-20);

  return (
    <div className="card">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Agent Activity</h3>
      <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
        {visible.length === 0 ? (
          <p className="text-xs text-gray-400 italic py-3 text-center">
            Waiting for updates...
          </p>
        ) : (
          visible.map((evt, i) => {
            const badge = agentBadgeColors[evt.agent_name] || 'bg-gray-100 text-gray-700';
            return (
              <div
                key={`${evt.timestamp}-${evt.agent_name}-${i}`}
                className="flex items-center gap-2 text-xs py-1 animate-slide-in-left"
              >
                <span className="text-gray-400 font-mono shrink-0">
                  {formatTime(evt.timestamp)}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded-full font-semibold shrink-0 ${badge}`}
                >
                  {evt.agent_name.replace(/_/g, ' ')}
                </span>
                <span className="text-gray-600 truncate">{actionText(evt)}</span>
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
