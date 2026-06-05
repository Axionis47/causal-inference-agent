import { useEffect, useMemo, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getJob, cancelJob, getNotebookUrl, AgentEvent, JobDetail } from '../services/api';
import { JOB_DETAIL_POLL_INTERVAL_MS } from '../config/constants';
import { useJob } from '../hooks/useJob';
import {
  SPECIALIST_ROSTER,
  deriveAgentStatuses,
  AgentTone,
} from '../components/job/terminal/agents';

// ─── small visual atoms ──────────────────────────────────────────────────

function StatusDot({ tone }: { tone: AgentTone }) {
  const cls: Record<AgentTone, string> = {
    pending: 'bg-edge-subtle',
    live: 'bg-amber animate-pulse-live',
    ok: 'bg-mint',
    failed: 'bg-rose',
  };
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${cls[tone]}`} />;
}

function Caption({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em] px-3 py-2 border-b border-edge-subtle">
      {children}
    </div>
  );
}

// ─── derived state helpers ───────────────────────────────────────────────

function formatElapsed(startISO: string | undefined, nowMs: number): string {
  if (!startISO) return '--:--:--';
  const startMs = Date.parse(startISO);
  if (Number.isNaN(startMs)) return '--:--:--';
  const sec = Math.max(0, Math.floor((nowMs - startMs) / 1000));
  const h = String(Math.floor(sec / 3600)).padStart(2, '0');
  const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function formatHMS(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '--:--:--';
  return d.toTimeString().slice(0, 8);
}

function formatDelta(curISO: string, priorISO: string | undefined): string {
  if (!priorISO) return '';
  const cur = Date.parse(curISO);
  const prior = Date.parse(priorISO);
  if (Number.isNaN(cur) || Number.isNaN(prior) || cur < prior) return '';
  const sec = Math.floor((cur - prior) / 1000);
  if (sec < 60) return `+${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `+${m}m${s.toString().padStart(2, '0')}`;
}

function statusTone(status: string | undefined): AgentTone {
  if (status === 'completed') return 'ok';
  if (status === 'failed') return 'failed';
  if (status === 'cancelled' || status === 'cancelling') return 'pending';
  return 'live';
}

function statusPillLabel(status: string | undefined): string {
  if (!status) return '----';
  if (status === 'completed') return 'DONE';
  if (status === 'failed') return 'FAIL';
  if (status === 'cancelled') return 'STOP';
  if (status === 'cancelling') return 'STOP…';
  return 'LIVE';
}

function describeEvent(ev: AgentEvent): string {
  const t = ev.event_type;
  const d = ev.data ?? {};
  if (typeof d.headline === 'string') return d.headline;
  if (t === 'agent_started') return `${ev.agent_name ?? 'agent'} dispatched`;
  if (t === 'agent_completed') return `${ev.agent_name ?? 'agent'} done`;
  if (typeof d.status === 'string') return `${t} · ${d.status}`;
  return t;
}

// ─── mock data for /jobs/__preview ───────────────────────────────────────

function buildPreviewState(): { job: JobDetail; events: AgentEvent[] } {
  const now = Date.now();
  const stamp = (offsetSec: number) =>
    new Date(now - offsetSec * 1000).toISOString();

  const events: AgentEvent[] = [
    { timestamp: stamp(240), agent_name: 'dataset_inspector', event_type: 'agent_started', data: {} },
    { timestamp: stamp(232), agent_name: 'dataset_inspector', event_type: 'agent_completed', data: { headline: 'picked train.csv · 200k rows · 18 cols' } },
    { timestamp: stamp(228), agent_name: 'data_profiler', event_type: 'agent_started', data: {} },
    { timestamp: stamp(210), agent_name: 'data_profiler', event_type: 'agent_completed', data: { headline: 'treatment=treatment, outcome=re78, 3 confounder candidates' } },
    { timestamp: stamp(208), agent_name: 'domain_knowledge', event_type: 'agent_started', data: {} },
    { timestamp: stamp(195), agent_name: 'domain_knowledge', event_type: 'agent_completed', data: { headline: 'labour-economics priors loaded' } },
    { timestamp: stamp(193), agent_name: 'eda', event_type: 'agent_started', data: {} },
    { timestamp: stamp(170), agent_name: 'eda', event_type: 'agent_completed', data: { headline: '2 confounders flagged: age, education' } },
    { timestamp: stamp(168), agent_name: 'causal_discovery', event_type: 'agent_started', data: {} },
    { timestamp: stamp(150), agent_name: 'causal_discovery', event_type: 'agent_completed', data: { headline: 'NOTEARS DAG, 11 edges' } },
    { timestamp: stamp(148), agent_name: 'dag_expert', event_type: 'agent_started', data: {} },
    { timestamp: stamp(130), agent_name: 'dag_expert', event_type: 'agent_completed', data: { headline: 'adjustment set: {age, educ, married}' } },
    { timestamp: stamp(128), agent_name: 'ps_diagnostics', event_type: 'agent_started', data: {} },
    { timestamp: stamp(110), agent_name: 'ps_diagnostics', event_type: 'agent_completed', data: { headline: 'overlap ok · SMD < 0.1 on 8/9 covariates' } },
    { timestamp: stamp(108), agent_name: 'effect_estimator', event_type: 'agent_started', data: {} },
    { timestamp: stamp(72),  agent_name: 'effect_estimator', event_type: 'agent_started', data: { headline: 'tool: get_dag_adjustment_set' } },
    { timestamp: stamp(28),  agent_name: 'effect_estimator', event_type: 'agent_started', data: { headline: 'DML running · 2/3 methods done' } },
  ];

  const job: JobDetail = {
    id: '7a3f1c2e-preview',
    status: 'estimating_effects',
    progress_percentage: 64,
    current_agent: 'effect_estimator',
    kaggle_url: 'kaggle://lalonde-nsw',
    treatment_variable: 'treatment',
    outcome_variable: 're78',
    iteration_count: 1,
    created_at: stamp(248),
    updated_at: stamp(0),
  };

  return { job, events };
}

// ─── page ────────────────────────────────────────────────────────────────

export default function JobPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const isPreview = jobId === '__preview';

  // Mount the .terminal class on <body> so global terminal CSS applies; clean up on unmount.
  useEffect(() => {
    document.body.classList.add('terminal');
    return () => document.body.classList.remove('terminal');
  }, []);

  // Live clock for the elapsed counter.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  // Real data path. useJob handles SSE; useQuery handles initial fetch + polling.
  const realStream = useJob(isPreview ? null : (jobId ?? null));
  const realJobQuery = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId && !isPreview,
    refetchInterval: (q: { state: { data?: { status?: string } } }) => {
      const s = q.state.data?.status;
      const terminal = s === 'completed' || s === 'failed' || s === 'cancelled';
      return terminal ? false : JOB_DETAIL_POLL_INTERVAL_MS;
    },
  });

  // Preview data path.
  const preview = useMemo(() => (isPreview ? buildPreviewState() : null), [isPreview]);

  const job: JobDetail | null = isPreview ? preview!.job : (realJobQuery.data ?? null);
  const agentEvents: AgentEvent[] = isPreview ? preview!.events : realStream.agentEvents;

  const cancelMutation = useMutation({
    mutationFn: () => cancelJob(jobId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  // F-keys.
  useEffect(() => {
    if (!job) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'F1' && job.status !== 'completed' && job.status !== 'failed' && !isPreview) {
        e.preventDefault();
        cancelMutation.mutate();
      }
      if (e.key === 'F3' && job.status === 'completed') {
        e.preventDefault();
        window.location.href = getNotebookUrl(job.id);
      }
      if (e.key === 'F5' && job.status === 'completed') {
        e.preventDefault();
        navigate(`/jobs/${job.id}#results`);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [job, cancelMutation, navigate, isPreview]);

  // Loading state.
  if (!isPreview && realJobQuery.isLoading) {
    return (
      <div className="terminal flex items-center justify-center min-h-screen">
        <span className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">connecting…</span>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="terminal flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em] mb-2">job not found</p>
          <button onClick={() => navigate('/jobs')} className="text-xs font-mono text-amber underline underline-offset-2">return to jobs list</button>
        </div>
      </div>
    );
  }

  // Derive what each pane needs.
  const completedAgents = new Set(
    agentEvents.filter(e => e.event_type === 'agent_completed').map(e => e.agent_name ?? ''),
  );
  const failed = job.status === 'failed';
  const agentTones = deriveAgentStatuses(job.current_agent, completedAgents, failed);
  const latestByAgent = new Map<string, AgentEvent>();
  for (const ev of agentEvents) {
    if (ev.agent_name) latestByAgent.set(ev.agent_name, ev);
  }
  const elapsed = formatElapsed(job.created_at, nowMs);
  const reverseEvents = [...agentEvents].reverse();
  const focusAgent = job.current_agent ?? null;
  const focusLatest = focusAgent ? latestByAgent.get(focusAgent) : undefined;

  return (
    <div className="terminal flex flex-col h-screen w-screen overflow-hidden bg-canvas text-ink">
      {/* ─── TOP BAR ────────────────────────────────────────────────── */}
      <header className="flex items-center h-10 px-3 bg-canvas-raised border-b border-edge-subtle shrink-0 gap-6">
        <div className="flex items-center gap-4 min-w-0 shrink-0">
          <span className="font-mono text-xs text-ink tabular">
            [ job <span className="text-ink-secondary">{job.id.slice(0, 8)}</span> ]
          </span>
          <span className="text-2xs font-mono uppercase tracking-[0.15em] text-indigo">
            standard
          </span>
        </div>

        {/* phase strip — one cell per specialist, lit by current tone. */}
        <div className="flex-1 flex items-center justify-center gap-0.5 min-w-0">
          {SPECIALIST_ROSTER.map(row => {
            const tone = agentTones[row.key];
            const bg =
              tone === 'live' ? 'bg-amber animate-pulse-live' :
              tone === 'ok' ? 'bg-mint' :
              tone === 'failed' ? 'bg-rose' :
              'bg-edge-subtle';
            return (
              <div
                key={row.key}
                className={`h-1.5 w-6 ${bg}`}
                title={`${row.label} · ${tone}`}
              />
            );
          })}
        </div>

        <div className="flex items-center gap-4 text-xs font-mono tabular shrink-0">
          <span className="inline-flex items-center gap-1.5">
            <StatusDot tone={statusTone(job.status)} />
            <span className="text-2xs uppercase tracking-[0.15em] text-ink-secondary">
              {statusPillLabel(job.status)}
            </span>
          </span>
          <span className="text-ink-secondary">
            <span className="text-ink-tertiary mr-1">elapsed</span>
            <span className="text-ink">{elapsed}</span>
          </span>
          <span className="text-ink-secondary">
            <span className="text-ink-tertiary mr-1">tokens</span>
            <span className="text-ink">—</span>
          </span>
          {!isPreview && job.status !== 'completed' && job.status !== 'failed' && (
            <button
              onClick={() => cancelMutation.mutate()}
              disabled={cancelMutation.isPending}
              className="text-2xs font-mono uppercase tracking-[0.15em] text-rose hover:text-ink border border-rose/40 hover:border-rose px-2 py-1 transition-colors"
            >
              {cancelMutation.isPending ? 'stopping…' : 'cancel'}
            </button>
          )}
        </div>
      </header>

      {/* ─── THREE PANES ────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* AGENTS */}
        <aside className="w-[260px] shrink-0 bg-canvas-raised border-r border-edge-subtle flex flex-col overflow-hidden">
          <Caption>[ agents ]</Caption>
          <div className="flex-1 overflow-y-auto">
            {SPECIALIST_ROSTER.map(row => {
              const tone = agentTones[row.key];
              const latest = latestByAgent.get(row.key);
              const headline = latest
                ? (typeof latest.data?.headline === 'string' ? latest.data.headline : describeEvent(latest))
                : row.answers;
              return (
                <div
                  key={row.key}
                  className={`flex items-center gap-2 px-3 py-2 border-b border-edge-subtle/40 ${
                    tone === 'live' ? 'bg-amber/5' : ''
                  }`}
                >
                  <StatusDot tone={tone} />
                  <div className="min-w-0 flex-1">
                    <div className={`font-mono text-xs truncate ${
                      tone === 'live' ? 'text-ink' :
                      tone === 'pending' ? 'text-ink-tertiary' :
                      'text-ink-secondary'
                    }`}>{row.label}</div>
                    <div className="font-mono text-2xs text-ink-tertiary truncate">{headline}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </aside>

        {/* TAPE */}
        <section className="flex-1 bg-canvas flex flex-col overflow-hidden min-w-0">
          <Caption>[ tape ]</Caption>
          <div className="flex-1 overflow-y-auto">
            {reverseEvents.length === 0 && (
              <div className="px-3 py-6 text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">
                awaiting first event…
              </div>
            )}
            {reverseEvents.map((ev, i) => {
              const key = `${ev.timestamp}-${ev.agent_name ?? '_'}-${ev.event_type}-${i}`;
              const isCurrent = ev.agent_name === job.current_agent;
              // reverseEvents[i+1] is the chronologically prior event.
              const priorISO = reverseEvents[i + 1]?.timestamp;
              const delta = formatDelta(ev.timestamp, priorISO);
              return (
                <div
                  key={key}
                  className="flex items-baseline gap-3 px-3 py-1.5 border-b border-edge-subtle/40 animate-tape-arrival"
                >
                  <span className="text-2xs font-mono text-ink-tertiary tabular shrink-0 w-16">
                    {formatHMS(ev.timestamp)}
                  </span>
                  <span className="shrink-0">
                    <StatusDot tone={isCurrent ? 'live' : 'ok'} />
                  </span>
                  <span className="font-mono text-xs shrink-0 w-44 truncate text-ink-secondary">
                    {ev.agent_name ?? 'orchestrator'}
                  </span>
                  <span className="font-mono text-xs text-bone flex-1 truncate min-w-0">
                    {describeEvent(ev)}
                  </span>
                  <span className="font-mono text-2xs text-ink-tertiary tabular shrink-0 w-16 text-right">
                    {delta}
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        {/* FOCUS */}
        <aside className="w-[300px] shrink-0 bg-canvas-raised border-l border-edge-subtle flex flex-col overflow-hidden">
          <Caption>[ focus ]</Caption>
          <div className="flex-1 overflow-y-auto p-3">
            {focusAgent ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <StatusDot tone={failed ? 'failed' : 'live'} />
                  <span className="font-mono text-base text-ink">{focusAgent}</span>
                </div>

                <dl className="space-y-1.5">
                  <FocusRow label="status" value={statusPillLabel(job.status).toLowerCase()} />
                  <FocusRow label="progress" value={`${job.progress_percentage ?? 0}%`} mono />
                  <FocusRow
                    label="last event"
                    value={focusLatest ? describeEvent(focusLatest) : 'awaiting…'}
                  />
                  <FocusRow label="iteration" value={String(job.iteration_count ?? 1)} mono />
                </dl>

                <div className="pt-2 border-t border-edge-subtle">
                  <div className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em] mb-2">
                    parameters
                  </div>
                  <dl className="space-y-1.5">
                    <FocusRow label="treatment" value={job.treatment_variable || 'auto'} mono />
                    <FocusRow label="outcome" value={job.outcome_variable || 'auto'} mono />
                    <FocusRow label="dataset" value={job.kaggle_url.replace(/^https?:\/\//, '')} mono truncate />
                  </dl>
                </div>

                {job.error_message && (
                  <div className="pt-2 border-t border-edge-subtle">
                    <div className="text-2xs font-mono text-rose uppercase tracking-[0.15em] mb-2">
                      error
                    </div>
                    <p className="text-xs font-mono text-bone">{job.error_message}</p>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">
                no agent currently dispatched
              </p>
            )}
          </div>
        </aside>
      </div>

      {/* ─── BOTTOM F-KEY BAR ──────────────────────────────────────── */}
      <footer className="flex items-center h-9 bg-canvas-raised border-t border-edge-subtle shrink-0 px-3 text-2xs font-mono">
        <FKey n="F1" label="cancel" enabled={!isPreview && job.status !== 'completed' && job.status !== 'failed'} onClick={() => cancelMutation.mutate()} />
        <FKey n="F2" label="approve" enabled={false} />
        <FKey n="F3" label="notebook" enabled={job.status === 'completed'} onClick={() => { window.location.href = getNotebookUrl(job.id); }} />
        <FKey n="F4" label="traces" enabled />
        <FKey n="F5" label="results" enabled={job.status === 'completed'} />
        <div className="ml-auto flex items-center gap-2 text-ink-tertiary">
          <span>{isPreview ? 'preview · synthetic data' : 'live · sse connected'}</span>
        </div>
      </footer>
    </div>
  );
}

// ─── tiny inline component helpers ───────────────────────────────────────

function FocusRow({
  label,
  value,
  mono,
  truncate,
}: { label: string; value: string; mono?: boolean; truncate?: boolean }) {
  return (
    <div className="flex items-baseline gap-3">
      <dt className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em] w-20 shrink-0">
        {label}
      </dt>
      <dd className={`text-xs ${mono ? 'font-mono' : ''} text-ink ${truncate ? 'truncate' : ''} min-w-0 flex-1`}>
        {value}
      </dd>
    </div>
  );
}

function FKey({
  n, label, enabled, onClick,
}: { n: string; label: string; enabled: boolean; onClick?: () => void }) {
  return (
    <button
      type="button"
      onClick={enabled ? onClick : undefined}
      disabled={!enabled}
      className={`flex items-center gap-1.5 px-3 py-1 border-r border-edge-subtle uppercase tracking-[0.15em] ${
        enabled ? 'text-ink-secondary hover:text-ink hover:bg-canvas-overlay' : 'text-ink-tertiary cursor-not-allowed'
      } transition-colors`}
    >
      <span className="text-ink-tertiary">{n}</span>
      <span>{label}</span>
    </button>
  );
}
