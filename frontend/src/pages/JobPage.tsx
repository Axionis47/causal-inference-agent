import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Download,
  StopCircle,
  Trash2,
} from 'lucide-react';
import { getJob, getResults, getNotebookUrl, cancelJob, deleteJob } from '../services/api';
import JobProgress from '../components/job/JobProgress';
import ActivityFeed from '../components/job/ActivityFeed';
import ResultsDisplay from '../components/results/ResultsDisplay';
import AgentTraces from '../components/agents/AgentTraces';
import { useJob } from '../hooks/useJob';

export default function JobPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // useJob hook for SSE-driven agent activity events
  const { agentEvents } = useJob(jobId ?? null);

  const jobQuery = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
    // Poll every 1 second while job is running to catch quick stage transitions
    refetchInterval: (query: { state: { data?: { status?: string } } }) => {
      const status = query.state.data?.status;
      const isTerminal = status === 'completed' || status === 'failed' || status === 'cancelled';
      return isTerminal ? false : 1000;
    },
  });

  const resultsQuery = useQuery({
    queryKey: ['results', jobId],
    queryFn: () => getResults(jobId!),
    enabled: jobQuery.data?.status === 'completed',
  });

  const cancelMutation = useMutation({
    mutationFn: () => cancelJob(jobId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (force: boolean) => deleteJob(jobId!, force),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      navigate('/jobs');
    },
  });

  if (jobQuery.isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="w-6 h-6 border-2 border-ink-900 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (jobQuery.isError || !jobQuery.data) {
    return (
      <div className="max-w-2xl mx-auto text-center py-16">
        <p className="font-serif text-xl text-ink-900 mb-2">Job Not Found</p>
        <p className="text-ink-500 text-sm">
          The job you are looking for does not exist or has been deleted.
        </p>
      </div>
    );
  }

  const job = jobQuery.data;
  const isComplete = job.status === 'completed';
  const isFailed = job.status === 'failed';
  const isCancelled = job.status === 'cancelled';
  const isRunning = !isComplete && !isFailed && !isCancelled;

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header: Job ID + Status */}
      <section className="section">
        <div className="flex items-start justify-between">
          <div>
            <p className="font-mono text-xs text-ink-400 tracking-wide mb-1">
              {job.id}
            </p>
            <h1 className="font-serif text-2xl text-ink-900">
              Causal Analysis
            </h1>
          </div>
          <StatusBadge status={job.status} />
        </div>
      </section>

      {/* Metadata */}
      <section className="section">
        <h2 className="section-title">Parameters</h2>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm">
          <div>
            <dt className="text-ink-400 mb-0.5">Dataset</dt>
            <dd className="font-mono text-xs text-ink-700 break-all">{job.kaggle_url}</dd>
          </div>
          <div>
            <dt className="text-ink-400 mb-0.5">Iterations</dt>
            <dd className="font-mono text-xs text-ink-700">{job.iteration_count}</dd>
          </div>
          <div>
            <dt className="text-ink-400 mb-0.5">Treatment Variable</dt>
            <dd className="font-mono text-xs text-ink-700">{job.treatment_variable || 'Auto-detected'}</dd>
          </div>
          <div>
            <dt className="text-ink-400 mb-0.5">Outcome Variable</dt>
            <dd className="font-mono text-xs text-ink-700">{job.outcome_variable || 'Auto-detected'}</dd>
          </div>
        </dl>
      </section>

      {/* Notebook download (completed only) */}
      {isComplete && (
        <section className="section">
          {resultsQuery.data?.notebook_url ? (
            <a
              href={getNotebookUrl(job.id)}
              download
              className="inline-flex items-center space-x-1.5 text-sm text-ink-600 hover:text-ink-900 underline underline-offset-2"
            >
              <Download className="w-4 h-4" />
              <span>Download Jupyter Notebook</span>
            </a>
          ) : resultsQuery.isSuccess && !resultsQuery.data?.notebook_url && (
            <p className="text-sm text-ink-400 italic">
              Notebook not available for this analysis.
            </p>
          )}
        </section>
      )}

      {/* Error state */}
      {isFailed && job.error_message && (
        <section className="section">
          <div className="border border-ink-200 p-4">
            <p className="font-serif text-sm text-ink-900 mb-1">Analysis Failed</p>
            <p className="text-sm text-ink-600">{job.error_message}</p>
          </div>
        </section>
      )}

      {/* Cancelled state */}
      {isCancelled && (
        <section className="section">
          <div className="border border-ink-200 p-4">
            <p className="font-serif text-sm text-ink-900 mb-1">Job Cancelled</p>
            <p className="text-sm text-ink-600">{job.error_message || 'This job was cancelled by user request.'}</p>
          </div>
        </section>
      )}

      {/* Actions */}
      <section className="section">
        <div className="flex items-center space-x-3">
          {isRunning && (
            <button
              onClick={() => cancelMutation.mutate()}
              disabled={cancelMutation.isPending}
              className="btn-secondary inline-flex items-center space-x-1.5"
            >
              <StopCircle className="w-4 h-4" />
              <span>{cancelMutation.isPending ? 'Cancelling...' : 'Stop Job'}</span>
            </button>
          )}

          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="btn-danger inline-flex items-center space-x-1.5"
          >
            <Trash2 className="w-4 h-4" />
            <span>Delete Job</span>
          </button>
        </div>
      </section>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-job-title"
          onKeyDown={(e) => {
            if (e.key === 'Escape') setShowDeleteConfirm(false);
          }}
        >
          <div className="bg-white border border-ink-200 p-6 max-w-md w-full mx-4">
            <h3 id="delete-job-title" className="font-serif text-lg text-ink-900 mb-2">Delete Job</h3>
            <p className="text-sm text-ink-600 mb-6">
              This will permanently delete the job record, analysis results, and all associated data.
              {isRunning && ' The job will be cancelled first.'}
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="btn-secondary"
                autoFocus
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  deleteMutation.mutate(isRunning);
                  setShowDeleteConfirm(false);
                }}
                disabled={deleteMutation.isPending}
                className="btn-danger"
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Progress (running only) */}
      {isRunning && (
        <>
          <section className="section">
            <h2 className="section-title">Progress</h2>
            <JobProgress
              status={job.status}
              progress={job.progress_percentage ?? 0}
              currentAgent={job.current_agent}
            />
          </section>
          <section className="section">
            <h2 className="section-title">Activity</h2>
            <ActivityFeed events={agentEvents} />
          </section>
        </>
      )}

      {/* Results (completed only) */}
      {isComplete && (
        <>
          {resultsQuery.isLoading && (
            <section className="section">
              <div className="flex items-center justify-center py-12">
                <div className="w-6 h-6 border-2 border-ink-900 border-t-transparent rounded-full animate-spin" />
                <span className="ml-3 text-sm text-ink-500">Loading analysis results...</span>
              </div>
            </section>
          )}
          {resultsQuery.isError && (
            <section className="section">
              <div className="border border-ink-200 p-4">
                <p className="font-serif text-sm text-ink-900 mb-1">Failed to load results</p>
                <p className="text-sm text-ink-500">Please try refreshing the page or check back later.</p>
              </div>
            </section>
          )}
          {resultsQuery.data && (
            <ResultsDisplay results={resultsQuery.data} />
          )}
        </>
      )}

      {/* Agent Traces */}
      {(isComplete || isFailed) && (
        <section className="section">
          <AgentTraces jobId={job.id} />
        </section>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { label: string; className: string }> = {
    pending: {
      label: 'Pending',
      className: 'border-ink-300 text-ink-500',
    },
    completed: {
      label: 'Completed',
      className: 'border-ink-900 text-ink-900',
    },
    failed: {
      label: 'Failed',
      className: 'border-ink-900 text-ink-900',
    },
    cancelled: {
      label: 'Cancelled',
      className: 'border-ink-400 text-ink-500',
    },
    cancelling: {
      label: 'Cancelling',
      className: 'border-ink-400 text-ink-500',
    },
  };

  const defaultConfig = {
    label: status.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    className: 'border-ink-900 text-ink-900',
  };

  const { label, className } = config[status] || defaultConfig;

  return (
    <span className={`inline-block border px-2 py-0.5 text-xs font-mono tracking-wide uppercase ${className}`}>
      {label}
    </span>
  );
}
