// JobPage — terminal-layout orchestrator.
//
// Responsibilities (and only these):
//   1. Read jobId from the route. Branch real-data vs `/jobs/__preview` mock.
//   2. Drive data: useJob (SSE) + useQuery (initial fetch + polling).
//   3. Manage the .terminal body class and the live elapsed-clock tick.
//   4. Wire keyboard shortcuts (F1 cancel / F3 notebook / F5 results).
//   5. Hand derived view state + callbacks to the five terminal panes.
//
// All presentation lives in components/job/terminal/.

import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getJob, getTraces, getGateSnapshot, cancelJob, getNotebookUrl, submitPlanDecision, AgentEvent, AgentTrace, JobDetail, DagGatePayload, PlanDecisionRequest, ResultsGatePayload, RelationalProfilePayload } from '../services/api';
import { JOB_DETAIL_POLL_INTERVAL_MS, TRACES_POLL_INTERVAL_MS } from '../config/constants';
import { useJob } from '../hooks/useJob';
import { deriveJobView } from '../components/job/terminal/deriveJobView';
import { tracesToEvents } from '../components/job/terminal/traceEvents';
import { ResultsView } from '../components/job/terminal/ResultsView';
import { resolveGate } from '../components/job/terminal/resolveGate';
import { TopBar } from '../components/job/terminal/TopBar';
import { AgentsRail } from '../components/job/terminal/AgentsRail';
import { Tape } from '../components/job/terminal/Tape';
import { FocusPane } from '../components/job/terminal/FocusPane';
import { FKeyBar } from '../components/job/terminal/FKeyBar';
import { RunAnalysisBar } from '../components/job/terminal/RunAnalysisBar';
import { DatasetView } from '../components/job/terminal/DatasetView';
import { ApprovalBar } from '../components/job/terminal/ApprovalBar';
import { DagGate } from '../components/job/terminal/DagGate';
import { ResultsGate } from '../components/job/terminal/ResultsGate';
import { buildPreviewState } from '../components/job/terminal/preview';
import { AnalysisView } from '../components/job/analysis/AnalysisView';
import { useAnalysis } from '../hooks/useAnalysis';
import { useDatasetView } from '../hooks/useDatasetView';

export default function JobPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const isPreview = jobId === '__preview';

  // Scope terminal styles to this page only; journal routes keep their light surface.
  useEffect(() => {
    document.body.classList.add('terminal');
    return () => document.body.classList.remove('terminal');
  }, []);

  // 1Hz tick for the elapsed counter.
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

  // Traces carry per-agent token_usage; we sum them for the TopBar counter.
  // Poll alongside the job until it reaches a terminal state, then freeze.
  const tracesQuery = useQuery({
    queryKey: ['job-traces', jobId],
    queryFn: () => getTraces(jobId!),
    enabled: !!jobId && !isPreview,
    refetchInterval: () => {
      const s = realJobQuery.data?.status;
      const terminal = s === 'completed' || s === 'failed' || s === 'cancelled';
      return terminal ? false : TRACES_POLL_INTERVAL_MS;
    },
  });
  const tokens = useMemo<number | null>(() => {
    const traces = tracesQuery.data;
    if (!traces || traces.length === 0) return null;
    return traces.reduce(
      (sum, t) => sum + (t.token_usage?.input_tokens || 0) + (t.token_usage?.output_tokens || 0),
      0,
    );
  }, [tracesQuery.data]);

  // The gate snapshot, rehydrated from REST so the gate panel survives a
  // refresh / SSE drop / event-buffer eviction. Only fetched while the job is
  // parked; the live SSE gate event takes precedence when it is in the buffer.
  const gateSnapshotQuery = useQuery({
    queryKey: ['gate', jobId],
    queryFn: () => getGateSnapshot(jobId!),
    enabled: !!jobId && !isPreview && realJobQuery.data?.status === 'awaiting_approval',
    refetchInterval: () =>
      realJobQuery.data?.status === 'awaiting_approval' ? JOB_DETAIL_POLL_INTERVAL_MS : false,
  });

  // The analysis view for the new analysis slice. Enabled once the job enters
  // running_analysis / waiting_for_user / completed; 404 (no run yet, or a
  // legacy job) resolves to null. SSE analysis_* events invalidate the query.
  const { analysis, isLoading: analysisLoading } = useAnalysis(
    isPreview ? null : (jobId ?? null),
    realJobQuery.data?.status,
  );

  // Dataset view (F1). The analyst lands here on arrival so the raw data and
  // download status are the first thing seen; Esc / F1 toggles back to the
  // agent tape. useDatasetView polls /jobs/:id/dataset until the blocks settle.
  const [showDataset, setShowDataset] = useState(!isPreview);
  const { view: datasetView } = useDatasetView(isPreview ? null : (jobId ?? null));

  // Which agent the focus pane is pinned to. null follows the live agent; a
  // click pins any agent (even a finished one) so its findings stay inspectable.
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  // The results overlay (effects, sensitivity, notebook). There is no separate
  // route; it opens over the terminal. Auto-open once when a job is completed so
  // the analysis output is visible instead of just the trace tape.
  const [showResults, setShowResults] = useState(false);
  const resultsAutoOpened = useRef(false);

  // Memoise preview so timestamps freeze at first render of /jobs/__preview.
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

  // Plan-gate decision (POST /jobs/:id/plan). Confirm resumes the run, reject
  // fails it; either way the job and analysis views change, so invalidate both.
  // AnalysisView -> PlanGate stay presentational; this page owns the mutation.
  const planMutation = useMutation({
    mutationFn: (body: PlanDecisionRequest) => submitPlanDecision(jobId!, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['analysis', jobId] });
    },
  });

  // At AWAITING_APPROVAL the latest gate event tells which of the three gates we
  // are at: the data gate (ApprovalBar over the dataset view), the DAG gate, or
  // the results gate (each with its own panel and payload).
  const activeGate = useMemo<
    | { kind: 'data' }
    | { kind: 'dag'; payload: DagGatePayload }
    | { kind: 'results'; payload: ResultsGatePayload }
    | null
  >(() => {
    if (job?.status !== 'awaiting_approval') return null;
    const gateTypes = ['approval_required', 'dag_approval_required', 'results_approval_required'];
    const gateEvents = agentEvents.filter((e) => gateTypes.includes(e.event_type));
    if (gateEvents.length === 0) return null;
    const latest = gateEvents.reduce((a, b) => (a.timestamp >= b.timestamp ? a : b));
    if (latest.event_type === 'dag_approval_required')
      return { kind: 'dag', payload: latest.data as unknown as DagGatePayload };
    if (latest.event_type === 'results_approval_required')
      return { kind: 'results', payload: latest.data as unknown as ResultsGatePayload };
    return { kind: 'data' };
  }, [job?.status, agentEvents]);

  // The same gate, rehydrated from the REST snapshot. Used as the fallback when
  // the live SSE gate event is not in the buffer (refresh / SSE drop / eviction).
  const gateFromSnapshot = useMemo(
    () => resolveGate(gateSnapshotQuery.data),
    [gateSnapshotQuery.data],
  );

  // Live SSE gate wins; the REST snapshot fills in when the event is gone.
  const gate = activeGate ?? gateFromSnapshot;

  // The bundle's relational structure, reported by the inspector right after
  // download. Shown in the dataset view so the analyst sees how the files relate
  // while reviewing the data. Null for single-file bundles.
  const relational = useMemo<RelationalProfilePayload | null>(() => {
    const evts = agentEvents.filter((e) => e.event_type === 'dataset_inspection_complete');
    if (evts.length === 0) return null;
    const latest = evts.reduce((a, b) => (a.timestamp >= b.timestamp ? a : b));
    return (latest.data?.relational_profile as RelationalProfilePayload | null) ?? null;
  }, [agentEvents]);

  // The data-gate snapshot carries the same relational profile, so the bundle
  // structure block also survives a refresh at the data gate.
  const relationalFromSnapshot = useMemo<RelationalProfilePayload | null>(() => {
    const snap = gateSnapshotQuery.data;
    if (!snap || snap.kind !== 'data') return null;
    return (snap.payload as { relational?: RelationalProfilePayload | null }).relational ?? null;
  }, [gateSnapshotQuery.data]);
  const resolvedRelational = relational ?? relationalFromSnapshot;

  // The data gate reviews data in the dataset view; the DAG and results gates
  // have their own panels, so keep the dataset view closed there.
  useEffect(() => {
    if (job?.status === 'awaiting_approval') {
      setShowDataset(gate === null || gate.kind === 'data');
    }
  }, [job?.status, gate]);

  // Auto-open the results overlay once a job is completed, so the analysis output
  // is what you see, not an empty page. Only once: a manual close stays closed.
  // Jobs completed by the new analysis slice show AnalysisView in the main area
  // instead, so wait until the analysis query has settled to null (legacy job)
  // before auto-opening.
  useEffect(() => {
    if (
      !isPreview &&
      job?.status === 'completed' &&
      !resultsAutoOpened.current &&
      !analysisLoading &&
      analysis === null
    ) {
      resultsAutoOpened.current = true;
      setShowResults(true);
    }
  }, [isPreview, job?.status, analysis, analysisLoading]);

  // F-key shortcuts. Always run the hook; gate the actions inside.
  useEffect(() => {
    if (!job) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'F1') {
        e.preventDefault();
        setShowDataset((v) => !v);
      }
      if (e.key === 'F2' && job.status !== 'completed' && job.status !== 'failed' && !isPreview) {
        e.preventDefault();
        cancelMutation.mutate();
      }
      if (e.key === 'F3' && job.status === 'completed') {
        e.preventDefault();
        window.location.href = getNotebookUrl(job.id);
      }
      if (e.key === 'F4' && job.status === 'completed') {
        e.preventDefault();
        navigate(`/jobs/${job.id}#results`);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [job, cancelMutation, navigate, isPreview]);

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

  // Per-agent reasoning steps, also used to hydrate the panes when there is no
  // live SSE stream. Reuse the traces already polled for the token counter.
  // Preview supplies its own.
  const allTraces: AgentTrace[] = isPreview ? (preview!.traces ?? []) : (tracesQuery.data ?? []);

  // A running job feeds the panes from live SSE events; a completed or reloaded
  // job has none, so backfill from the persisted traces instead of leaving the
  // tape stuck on "awaiting first event".
  const sourceEvents = agentEvents.length > 0 ? agentEvents : tracesToEvents(allTraces);

  const view = deriveJobView(job, sourceEvents, nowMs, selectedAgent);
  const onCancel = () => cancelMutation.mutate();

  // The new analysis slice owns the main area while it runs or waits, and once
  // its view exists (which also covers completed runs). Legacy jobs, where
  // /analysis 404s to null, keep the three-pane tape layout.
  const showAnalysisArea =
    !isPreview &&
    (analysis !== null ||
      job.status === 'running_analysis' ||
      job.status === 'waiting_for_user');

  const focusTraces = view.focusAgent
    ? allTraces.filter((t) => t.agent_name === view.focusAgent)
    : [];

  // Advisory gate for the data-review confirm button: a non-empty causal
  // question must be set (and the time column, if the dataset has one, must be
  // a real column), mirroring the backend confirm guard
  // (manager._require_confirmable_dataset). The server stays authoritative;
  // this just blocks a confirm that would otherwise 422.
  const profileData = datasetView?.profile.data ?? null;
  const profileColumns = profileData ? Object.keys(profileData.feature_types ?? {}) : [];
  const hasTimeDim = profileData?.has_time_dimension ?? false;
  const timeCol = profileData?.time_column ?? '';
  const canConfirmData =
    !!profileData &&
    !!(job.causal_question && job.causal_question.trim()) &&
    (!hasTimeDim || (timeCol !== '' && profileColumns.includes(timeCol)));

  return (
    <div className="terminal flex flex-col h-screen w-screen overflow-hidden bg-canvas text-ink">
      <TopBar
        job={job}
        agentTones={view.agentTones}
        elapsed={view.elapsed}
        tokens={isPreview ? null : tokens}
        isPreview={isPreview}
        onCancel={onCancel}
        cancelPending={cancelMutation.isPending}
      />

      <div className="flex-1 flex overflow-hidden min-h-0">
        {showAnalysisArea ? (
          analysis ? (
            <AnalysisView
              analysis={analysis}
              onPlanConfirm={(edits) =>
                planMutation.mutate({
                  decision: 'confirm',
                  ...(Object.keys(edits).length > 0 ? { edits } : {}),
                })
              }
              onPlanReject={(reason) => planMutation.mutate({ decision: 'reject', reason })}
              planSubmitting={planMutation.isPending}
              planError={planMutation.error ? planMutation.error.message : null}
            />
          ) : (
            <div className="flex-1 flex items-center justify-center bg-canvas">
              <span className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">
                {analysisLoading ? 'loading analysis…' : 'analysis starting…'}
              </span>
            </div>
          )
        ) : (
          <>
            <AgentsRail
              tones={view.agentTones}
              latestByAgent={view.latestByAgent}
              findingByAgent={view.findingByAgent}
              challengedAgents={view.challengedAgents}
              selected={selectedAgent}
              onSelect={(key) => setSelectedAgent((prev) => (prev === key ? null : key))}
            />
            <Tape reverseEvents={view.reverseEvents} currentAgent={job.current_agent} />
            <FocusPane
              job={job}
              focusAgent={view.focusAgent}
              focusLatest={view.focusLatest}
              focusFinding={view.focusFinding}
              focusChallenge={view.focusChallenge}
              focusTraces={focusTraces}
              focusTone={view.focusAgent ? view.agentTones[view.focusAgent] : undefined}
              failed={view.failed}
              datasetView={datasetView}
              onOpenData={() => setShowDataset(true)}
            />
          </>
        )}
      </div>

      <FKeyBar
        job={job}
        isPreview={isPreview}
        onCancel={onCancel}
        onData={() => setShowDataset((v) => !v)}
        onResults={() => setShowResults((v) => !v)}
      />

      {showDataset && (
        <DatasetView
          view={datasetView}
          jobId={jobId ?? null}
          causalQuestion={job.causal_question ?? null}
          relational={resolvedRelational}
          onClose={() => setShowDataset(false)}
        />
      )}

      {!isPreview && job.status === 'awaiting_approval' && (
        gate?.kind === 'dag' ? (
          <DagGate jobId={job.id} payload={gate.payload} />
        ) : gate?.kind === 'results' ? (
          <ResultsGate jobId={job.id} payload={gate.payload} />
        ) : (
          <ApprovalBar
            jobId={job.id}
            canConfirm={canConfirmData}
            onOpenData={() => setShowDataset(true)}
          />
        )
      )}

      {!isPreview && job.status === 'confirmed' && (
        <RunAnalysisBar jobId={job.id} onOpenData={() => setShowDataset(true)} />
      )}

      {!isPreview && showResults && (
        <ResultsView jobId={job.id} onClose={() => setShowResults(false)} />
      )}
    </div>
  );
}
