import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { listJobs, Job, cancelJob, deleteJob } from '../services/api';
import toast from 'react-hot-toast';
import { DEFAULT_PAGE_SIZE, JOBS_LIST_REFRESH_INTERVAL_MS } from '../config/constants';

const PAGE_SIZE = DEFAULT_PAGE_SIZE;

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'pending', label: 'Pending' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'cancelled', label: 'Cancelled' },
];

/** Extract a human-readable dataset name from a Kaggle URL. */
function extractDatasetName(url: string): string {
  try {
    const parts = url.replace(/\/+$/, '').split('/');
    const name = parts[parts.length - 1] || parts[parts.length - 2] || url;
    return name;
  } catch {
    return url;
  }
}

/** Return a relative time string like "2 minutes ago". */
function relativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffSec = Math.floor((now - then) / 1000);

  if (diffSec < 60) return 'just now';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} minute${diffMin !== 1 ? 's' : ''} ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr} hour${diffHr !== 1 ? 's' : ''} ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay} day${diffDay !== 1 ? 's' : ''} ago`;
  const diffMon = Math.floor(diffDay / 30);
  return `${diffMon} month${diffMon !== 1 ? 's' : ''} ago`;
}

/** Format a duration in ms as a compact human string. */
function formatDuration(ms: number): string {
  if (ms < 1000) return '<1s';
  const sec = Math.round(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m`;
  const hr = (min / 60).toFixed(1);
  return `${hr}h`;
}

/** Job duration: terminal jobs use updated_at - created_at; else null. */
function jobDuration(job: Job): string | null {
  const isTerminal = ['completed', 'failed', 'cancelled'].includes(job.status);
  if (!isTerminal) return null;
  const ms = new Date(job.updated_at).getTime() - new Date(job.created_at).getTime();
  if (!isFinite(ms) || ms <= 0) return null;
  return formatDuration(ms);
}

/** A 6px status dot per aesthetics.md §6. */
function StatusDot({ tone }: { tone: string }) {
  const map: Record<string, string> = {
    live: 'bg-amber animate-pulse-live',
    ok: 'bg-mint',
    failed: 'bg-rose',
    cancelled: 'bg-ink-tertiary',
    pending: 'bg-edge-subtle',
  };
  return (
    <span
      className={`inline-block w-1.5 h-1.5 rounded-full ${map[tone] || map.pending}`}
      aria-hidden="true"
    />
  );
}

/** Status tone + label. Tone maps to a semantic dot colour. */
function statusDisplay(status: string): { tone: string; label: string } {
  const isRunning = !['completed', 'failed', 'pending', 'cancelled'].includes(status);
  if (isRunning) return { tone: 'live', label: status.replace('_', ' ') };

  switch (status) {
    case 'completed':
      return { tone: 'ok', label: 'done' };
    case 'failed':
      return { tone: 'failed', label: 'failed' };
    case 'cancelled':
      return { tone: 'cancelled', label: 'cancelled' };
    case 'pending':
    default:
      return { tone: 'pending', label: 'pending' };
  }
}

export default function JobsListPage() {
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState('');

  const jobsQuery = useQuery({
    queryKey: ['jobs', statusFilter, page],
    queryFn: () => listJobs(statusFilter || undefined, PAGE_SIZE, page * PAGE_SIZE),
    // Auto-refresh to catch new jobs
    refetchInterval: JOBS_LIST_REFRESH_INTERVAL_MS,
  });

  const jobs = jobsQuery.data?.jobs || [];
  const totalJobs = jobsQuery.data?.total || 0;
  const totalPages = Math.ceil(totalJobs / PAGE_SIZE);

  // Loading state — plain text
  if (jobsQuery.isLoading) {
    return (
      <div className="max-w-5xl mx-auto">
        <h1 className="text-xl font-semibold text-ink border-b border-edge pb-3 mb-6">
          Automated Analyses
        </h1>
        <p className="text-sm text-ink-secondary" aria-label="Loading jobs">Loading...</p>
      </div>
    );
  }

  // Error state — plain text, no card
  if (jobsQuery.isError) {
    return (
      <div className="max-w-5xl mx-auto">
        <h1 className="text-xl font-semibold text-ink border-b border-edge pb-3 mb-6">
          Automated Analyses
        </h1>
        <div role="alert">
          <p className="text-sm text-rose mb-2">
            Failed to load jobs: {(jobsQuery.error as { message?: string })?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={() => jobsQuery.refetch()}
            className="text-sm text-ink-secondary underline hover:text-ink"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header row: title + filter */}
      <div className="flex items-end justify-between border-b border-edge pb-3 mb-6">
        <h1 className="text-xl font-semibold text-ink">
          Automated Analyses
        </h1>
        <div className="flex items-center gap-3">
          {totalJobs > 0 && (
            <span className="text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary tabular">
              {totalJobs} job{totalJobs !== 1 ? 's' : ''}
            </span>
          )}
          <label htmlFor="status-filter" className="sr-only">Filter by status</label>
          <select
            id="status-filter"
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(0);
            }}
            className="font-mono text-xs text-ink bg-canvas-inset border border-edge-subtle px-3 py-1.5 focus:border-edge-strong transition-colors"
            aria-label="Filter jobs by status"
          >
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Empty state — plain text */}
      {jobs.length === 0 ? (
        <div className="py-12">
          <p className="text-sm text-ink-secondary mb-1">
            {statusFilter
              ? `No jobs found with status "${statusFilter}".`
              : 'No analyses yet.'}
          </p>
          {statusFilter ? (
            <p className="text-sm text-ink-tertiary">Try a different filter.</p>
          ) : (
            <Link to="/" className="text-sm text-indigo underline hover:text-ink">
              Start your first analysis
            </Link>
          )}
        </div>
      ) : (
        <>
          {/* Jobs table */}
          <table className="w-full text-sm" role="table" aria-label="Analysis jobs">
            <thead>
              <tr className="border-b border-edge">
                {['Status', 'Dataset', 'T → Y', 'Iter', 'Duration', 'Created', 'Actions'].map((h) => (
                  <th
                    key={h}
                    scope="col"
                    className={`py-2 px-3 font-mono text-2xs uppercase tracking-[0.12em] text-ink-tertiary ${
                      h === 'Iter' ? 'text-right' : 'text-left'
                    }`}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <JobRow key={job.id} job={job} />
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-end gap-4 mt-4 text-xs font-mono text-ink-secondary">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="hover:text-ink disabled:text-ink-tertiary disabled:cursor-not-allowed"
                aria-label="Previous page"
              >
                Previous
              </button>
              <span aria-live="polite" className="tabular">
                Page {page + 1} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="hover:text-ink disabled:text-ink-tertiary disabled:cursor-not-allowed"
                aria-label="Next page"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function JobRow({ job }: { job: Job }) {
  const queryClient = useQueryClient();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isRunning = !['completed', 'failed', 'pending', 'cancelled'].includes(job.status);

  const cancelMutation = useMutation({
    mutationFn: () => cancelJob(job.id),
    onSuccess: () => {
      setError(null);
      toast.success('Job cancelled');
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
    onError: (err: Error) => {
      setError(`Failed to cancel: ${err.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (force: boolean) => deleteJob(job.id, force),
    onSuccess: () => {
      setError(null);
      toast.success('Job deleted');
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
    onError: (err: Error) => {
      setError(`Failed to delete: ${err.message}`);
    },
  });

  const { tone, label } = statusDisplay(job.status);
  const datasetName = job.dataset_name || extractDatasetName(job.kaggle_url);
  const duration = jobDuration(job);
  const treatment = job.treatment_variable;
  const outcome = job.outcome_variable;
  const tyDisplay = treatment || outcome
    ? `${treatment || '—'} → ${outcome || '—'}`
    : null;
  const iter = job.iteration_count ?? 0;

  return (
    <>
      <tr className="group border-b border-edge-subtle hover:bg-canvas-raised">
        {/* Status */}
        <td className="py-2.5 px-3">
          <span className="inline-flex items-center gap-1.5">
            <StatusDot tone={tone} />
            <span className="text-2xs font-mono uppercase tracking-[0.12em] text-ink-secondary">
              {label}
            </span>
          </span>
        </td>

        {/* Dataset — name links to job detail; Kaggle URL on hover */}
        <td className="py-2.5 px-3">
          <Link
            to={`/jobs/${job.id}`}
            className="text-sm text-ink hover:text-mint hover:underline"
            title={job.kaggle_url}
          >
            {datasetName}
          </Link>
          <p className="font-mono text-2xs text-ink-tertiary tracking-wide tabular">
            {job.id.slice(0, 8)}
          </p>
        </td>

        {/* T → Y — inferred or specified */}
        <td className="py-2.5 px-3">
          {tyDisplay ? (
            <span className="font-mono text-xs text-ink-secondary">{tyDisplay}</span>
          ) : (
            <span className="text-xs text-ink-tertiary italic">auto-detect</span>
          )}
        </td>

        {/* Iteration count */}
        <td className="py-2.5 px-3 text-right">
          <span className="font-mono text-xs text-ink-secondary tabular">{iter}</span>
        </td>

        {/* Duration — terminal jobs only */}
        <td className="py-2.5 px-3 text-xs font-mono text-ink-tertiary tabular">
          {duration ?? <span className="text-ink-tertiary">—</span>}
        </td>

        {/* Created — relative time */}
        <td className="py-2.5 px-3 text-xs text-ink-tertiary">
          {relativeTime(job.created_at)}
        </td>

        {/* Actions — small text buttons */}
        <td className="py-2.5 px-3">
          <div className="flex items-center gap-3">
            {isRunning && (
              <button
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending}
                className="text-xs font-mono text-ink-secondary hover:text-ink disabled:text-ink-tertiary"
                aria-label={`Stop job ${job.id}`}
              >
                {cancelMutation.isPending ? 'Stopping...' : 'Stop'}
              </button>
            )}
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="text-xs font-mono text-rose hover:text-ink"
              aria-label={`Delete job ${job.id}`}
            >
              Delete
            </button>
          </div>
          {error && (
            <p className="text-xs text-rose mt-0.5">
              {error}{' '}
              <button onClick={() => setError(null)} className="underline">dismiss</button>
            </p>
          )}
        </td>
      </tr>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <tr>
          <td colSpan={7} className="!py-0">
            <div
              className="terminal fixed inset-0 bg-black/60 flex items-center justify-center z-50"
              role="dialog"
              aria-modal="true"
              aria-labelledby={`delete-title-${job.id}`}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setShowDeleteConfirm(false);
              }}
            >
              <div className="bg-canvas-overlay border border-edge-strong rounded-md p-6 max-w-md w-full mx-4">
                <h3 id={`delete-title-${job.id}`} className="text-base font-semibold text-ink mb-2">
                  Delete Job <span className="font-mono">{job.id.slice(0, 8)}</span>?
                </h3>
                <p className="text-sm text-ink-secondary mb-4">
                  This will permanently delete the job record, analysis results, and all associated data.
                  {isRunning && ' The job will be cancelled first.'}
                </p>
                <div className="flex justify-end gap-3 items-center">
                  <button
                    onClick={() => setShowDeleteConfirm(false)}
                    className="text-sm text-ink-secondary hover:text-ink"
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
                    className="bg-rose text-ink-inverse rounded-md px-3 py-1.5 text-sm font-medium hover:bg-rose-dim disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
