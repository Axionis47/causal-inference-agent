/**
 * ReviewTimeline — journal-style chronological merge of agent_traces
 * and decision_log so a user can read the run as a narrative rather
 * than a debug log.
 *
 * Pulls from existing endpoints (no backend change): getTraces() for
 * the trace stream and decision_log already on the results payload.
 * Merges by timestamp, sorts ascending, renders each entry with a
 * clear lead (who did what) and a paragraph of rationale below.
 */

import { memo, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, ChevronRight } from 'lucide-react';
import {
  AgentTrace,
  MethodologyDecision,
  getTraces,
} from '../../services/api';

interface ReviewTimelineProps {
  jobId: string;
  decisions: MethodologyDecision[];
}

type Entry =
  | { kind: 'decision'; ts: number; decision: MethodologyDecision }
  | { kind: 'trace'; ts: number; trace: AgentTrace };

type FilterKind = 'all' | 'decisions' | 'traces';

const AGENT_TONE: Record<string, string> = {
  orchestrator: 'text-violet-700 bg-violet-50 border-violet-200',
  data_profiler: 'text-blue-700 bg-blue-50 border-blue-200',
  causal_discovery: 'text-emerald-700 bg-emerald-50 border-emerald-200',
  effect_estimator: 'text-amber-700 bg-amber-50 border-amber-200',
  effect_estimator_react: 'text-amber-700 bg-amber-50 border-amber-200',
  sensitivity_analyst: 'text-orange-700 bg-orange-50 border-orange-200',
  critique: 'text-rose-700 bg-rose-50 border-rose-200',
  notebook_generator: 'text-indigo-700 bg-indigo-50 border-indigo-200',
  dag_expert: 'text-teal-700 bg-teal-50 border-teal-200',
  confounder_discovery: 'text-cyan-700 bg-cyan-50 border-cyan-200',
  ps_diagnostics: 'text-lime-700 bg-lime-50 border-lime-200',
  data_repair: 'text-rose-700 bg-rose-50 border-rose-200',
  domain_knowledge: 'text-violet-700 bg-violet-50 border-violet-200',
  eda_agent: 'text-sky-700 bg-sky-50 border-sky-200',
};

function agentTone(name: string): string {
  return AGENT_TONE[name] || 'text-ink-700 bg-ink-50 border-ink-200';
}

function timestampMs(s: string): number {
  const t = Date.parse(s);
  return isFinite(t) ? t : 0;
}

function formatClock(s: string): string {
  try {
    return new Date(s).toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return s;
  }
}

function ReviewTimelineImpl({ jobId, decisions }: ReviewTimelineProps) {
  const [filter, setFilter] = useState<FilterKind>('all');
  const [agentFilter, setAgentFilter] = useState<string>('all');

  const tracesQuery = useQuery({
    queryKey: ['traces', jobId],
    queryFn: () => getTraces(jobId),
  });

  const entries: Entry[] = useMemo(() => {
    const merged: Entry[] = [];
    for (const d of decisions) {
      merged.push({ kind: 'decision', ts: timestampMs(d.timestamp), decision: d });
    }
    for (const t of tracesQuery.data ?? []) {
      merged.push({ kind: 'trace', ts: timestampMs(t.timestamp), trace: t });
    }
    merged.sort((a, b) => a.ts - b.ts);
    return merged;
  }, [decisions, tracesQuery.data]);

  const uniqueAgents = useMemo(() => {
    const set = new Set<string>();
    for (const e of entries) {
      set.add(e.kind === 'decision' ? e.decision.agent : e.trace.agent_name);
    }
    return Array.from(set).sort();
  }, [entries]);

  const visible = useMemo(() => {
    return entries.filter((e) => {
      if (filter === 'decisions' && e.kind !== 'decision') return false;
      if (filter === 'traces' && e.kind !== 'trace') return false;
      const agent = e.kind === 'decision' ? e.decision.agent : e.trace.agent_name;
      if (agentFilter !== 'all' && agent !== agentFilter) return false;
      return true;
    });
  }, [entries, filter, agentFilter]);

  const counts = useMemo(
    () => ({
      decisions: entries.filter((e) => e.kind === 'decision').length,
      traces: entries.filter((e) => e.kind === 'trace').length,
    }),
    [entries]
  );

  if (tracesQuery.isLoading && decisions.length === 0) {
    return (
      <p className="text-sm text-ink-400 italic">Loading review…</p>
    );
  }

  if (entries.length === 0) {
    return (
      <p className="text-sm text-ink-400 italic">
        No traces or decisions recorded yet.
      </p>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-4 text-xs">
        <FilterChip
          label={`All · ${entries.length}`}
          active={filter === 'all'}
          onClick={() => setFilter('all')}
        />
        <FilterChip
          label={`Decisions · ${counts.decisions}`}
          active={filter === 'decisions'}
          onClick={() => setFilter('decisions')}
        />
        <FilterChip
          label={`Traces · ${counts.traces}`}
          active={filter === 'traces'}
          onClick={() => setFilter('traces')}
        />
        {uniqueAgents.length > 1 && (
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value)}
            className="ml-auto text-xs border border-ink-200 px-2 py-1 bg-white"
            aria-label="Filter by agent"
          >
            <option value="all">All agents</option>
            {uniqueAgents.map((a) => (
              <option key={a} value={a}>
                {a.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        )}
      </div>

      <ol className="space-y-3">
        {visible.map((e, i) =>
          e.kind === 'decision' ? (
            <DecisionEntry key={`d-${i}-${e.ts}`} decision={e.decision} />
          ) : (
            <TraceEntry key={`t-${i}-${e.ts}`} trace={e.trace} />
          )
        )}
      </ol>
    </div>
  );
}

interface FilterChipProps {
  label: string;
  active: boolean;
  onClick: () => void;
}
function FilterChip({ label, active, onClick }: FilterChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        active
          ? 'px-2.5 py-1 border border-ink-900 bg-ink-900 text-white font-medium'
          : 'px-2.5 py-1 border border-ink-200 text-ink-600 hover:border-ink-400'
      }
    >
      {label}
    </button>
  );
}

function DecisionEntry({ decision }: { decision: MethodologyDecision }) {
  const tone = agentTone(decision.agent);
  const hasAlternatives = (decision.alternatives?.length ?? 0) > 0;
  const [expanded, setExpanded] = useState(false);
  return (
    <li className="flex animate-fade-in">
      <div className="flex-none w-20 pt-1">
        <p className="font-mono text-[10px] text-ink-400 leading-none">
          {formatClock(decision.timestamp)}
        </p>
        <span
          aria-hidden="true"
          className="mt-1 inline-block w-2 h-2 rounded-full bg-ink-700"
        />
      </div>
      <div className="flex-1 min-w-0 border-l border-ink-100 pl-4">
        <p className="text-sm">
          <span className={`badge ${tone} mr-2`}>{decision.agent}</span>
          <span className="font-medium text-ink-900">decided</span>
          <span className="text-ink-700"> {decision.choice}</span>
          <span className="ml-2 text-[10px] uppercase tracking-wider text-ink-400">
            {decision.decision_type}
          </span>
        </p>
        {decision.reason && (
          <p className="text-sm text-ink-600 mt-1 leading-relaxed">
            {decision.reason}
          </p>
        )}
        {hasAlternatives && (
          <>
            <button
              type="button"
              onClick={() => setExpanded((e) => !e)}
              className="text-xs text-ink-500 hover:text-ink-900 mt-1.5 flex items-center"
            >
              {expanded ? (
                <ChevronDown className="w-3 h-3 mr-1" />
              ) : (
                <ChevronRight className="w-3 h-3 mr-1" />
              )}
              {decision.alternatives!.length} alternative
              {decision.alternatives!.length === 1 ? '' : 's'} considered
            </button>
            {expanded && (
              <ul className="mt-2 ml-4 space-y-1">
                {decision.alternatives!.map((alt, i) => (
                  <li key={i} className="text-xs text-ink-500">
                    <span className="font-mono text-ink-700">{alt.option}</span>
                    <span> — {alt.reason}</span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </li>
  );
}

function TraceEntry({ trace }: { trace: AgentTrace }) {
  const tone = agentTone(trace.agent_name);
  const reasoning = (trace.reasoning ?? '').trim();
  return (
    <li className="flex animate-fade-in">
      <div className="flex-none w-20 pt-1">
        <p className="font-mono text-[10px] text-ink-400 leading-none">
          {formatClock(trace.timestamp)}
        </p>
        <span
          aria-hidden="true"
          className="mt-1 inline-block w-2 h-2 rounded-full border border-ink-400"
        />
      </div>
      <div className="flex-1 min-w-0 border-l border-ink-100 pl-4">
        <p className="text-sm flex items-baseline gap-2">
          <span className={`badge ${tone}`}>{trace.agent_name}</span>
          <span className="text-ink-700">{trace.action}</span>
          <span className="ml-auto text-[10px] font-mono text-ink-400">
            {formatDuration(trace.duration_ms)}
          </span>
        </p>
        {reasoning && (
          <p className="text-sm text-ink-600 mt-1 leading-relaxed">
            {reasoning}
          </p>
        )}
        {trace.tools_called.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {trace.tools_called.map((tool, i) => (
              <span
                key={i}
                className="text-[10px] font-mono text-ink-500 bg-ink-50 border border-ink-100 px-1.5 py-0.5"
              >
                {tool}
              </span>
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

function formatDuration(ms: number): string {
  if (!ms || ms < 0) return '';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export default memo(ReviewTimelineImpl);
