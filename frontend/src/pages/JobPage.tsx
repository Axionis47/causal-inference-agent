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

import { useEffect, useMemo, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getJob, cancelJob, getNotebookUrl, AgentEvent, JobDetail } from '../services/api';
import { JOB_DETAIL_POLL_INTERVAL_MS } from '../config/constants';
import { useJob } from '../hooks/useJob';
import { deriveJobView } from '../components/job/terminal/deriveJobView';
import { TopBar } from '../components/job/terminal/TopBar';
import { AgentsRail } from '../components/job/terminal/AgentsRail';
import { Tape } from '../components/job/terminal/Tape';
import { FocusPane } from '../components/job/terminal/FocusPane';
import { FKeyBar } from '../components/job/terminal/FKeyBar';
import { buildPreviewState } from '../components/job/terminal/preview';

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

  // F-key shortcuts. Always run the hook; gate the actions inside.
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

  const view = deriveJobView(job, agentEvents, nowMs);
  const onCancel = () => cancelMutation.mutate();

  return (
    <div className="terminal flex flex-col h-screen w-screen overflow-hidden bg-canvas text-ink">
      <TopBar
        job={job}
        agentTones={view.agentTones}
        elapsed={view.elapsed}
        isPreview={isPreview}
        onCancel={onCancel}
        cancelPending={cancelMutation.isPending}
      />

      <div className="flex-1 flex overflow-hidden min-h-0">
        <AgentsRail tones={view.agentTones} latestByAgent={view.latestByAgent} />
        <Tape reverseEvents={view.reverseEvents} currentAgent={job.current_agent} />
        <FocusPane
          job={job}
          focusAgent={view.focusAgent}
          focusLatest={view.focusLatest}
          failed={view.failed}
        />
      </div>

      <FKeyBar job={job} isPreview={isPreview} onCancel={onCancel} />
    </div>
  );
}
