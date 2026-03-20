import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { listJobs, Job, cancelJob, deleteJob } from '../services/api';
import toast from 'react-hot-toast';

const PAGE_SIZE = 20;

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

/** Status indicator config. */
function statusDisplay(status: string) {
  const isRunning = !['completed', 'failed', 'pending', 'cancelled'].includes(status);

  if (isRunning) {
    return {
      indicator: (
        <span
          className="inline-block w-2 h-2 bg-blue-500 mr-2 pulse-dot"
          aria-hidden="true"
        />
      ),
      label: status.replace('_', ' '),
      badgeClass: 'bg-blue-50 text-blue-700 border border-blue-300',
    };
  }

  switch (status) {
    case 'completed':
      return {
        indicator: <span className="mr-1.5 text-green-600" aria-hidden="true">&check;</span>,
        label: 'done',
        badgeClass: 'bg-green-50 text-green-700 border border-green-300',
      };
    case 'failed':
      return {
        indicator: <span className="mr-1.5 text-red-600" aria-hidden="true">&times;</span>,
        label: 'failed',
        badgeClass: 'bg-red-50 text-red-700 border border-red-300',
      };
    case 'cancelled':
      return {
        indicator: <span className="mr-1.5 text-gray-500" aria-hidden="true">&mdash;</span>,
        label: 'cancelled',
        badgeClass: 'bg-gray-50 text-gray-600 border border-gray-300',
      };
    case 'pending':
    default:
      return {
        indicator: <span className="inline-block w-2 h-2 bg-gray-400 mr-2" aria-hidden="true" />,
        label: 'pending',
        badgeClass: 'bg-gray-50 text-gray-600 border border-gray-300',
      };
  }
}

export default function JobsListPage() {
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState('');

  const jobsQuery = useQuery({
    queryKey: ['jobs', statusFilter, page],
    queryFn: () => listJobs(statusFilter || undefined, PAGE_SIZE, page * PAGE_SIZE),
    // Auto-refresh every 10 seconds to catch new jobs
    refetchInterval: 10000,
  });

  const jobs = jobsQuery.data?.jobs || [];
  const totalJobs = jobsQuery.data?.total || 0;
  const totalPages = Math.ceil(totalJobs / PAGE_SIZE);

  // Loading state — plain text
  if (jobsQuery.isLoading) {
    return (
      <div className="max-w-5xl mx-auto">
        <h1 className="font-serif text-2xl font-bold text-ink-900 border-b border-ink-200 pb-3 mb-6">
          Automated Analyses
        </h1>
        <p className="text-sm text-ink-500" aria-label="Loading jobs">Loading...</p>
      </div>
    );
  }

  // Error state — plain text, no card
  if (jobsQuery.isError) {
    return (
      <div className="max-w-5xl mx-auto">
        <h1 className="font-serif text-2xl font-bold text-ink-900 border-b border-ink-200 pb-3 mb-6">
          Automated Analyses
        </h1>
        <div role="alert">
          <p className="text-sm text-red-700 mb-2">
            Failed to load jobs: {(jobsQuery.error as { message?: string })?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={() => jobsQuery.refetch()}
            className="text-sm text-ink-700 underline hover:text-ink-900"
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
      <div className="flex items-end justify-between border-b border-ink-200 pb-3 mb-6">
        <h1 className="font-serif text-2xl font-bold text-ink-900">
          Automated Analyses
        </h1>
        <div className="flex items-center gap-3">
          {totalJobs > 0 && (
            <span className="text-xs text-ink-400">
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
            className="input-field !w-auto !py-1.5 !px-3 text-xs"
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
          <p className="text-sm text-ink-500 mb-1">
            {statusFilter
              ? `No jobs found with status "${statusFilter}".`
              : 'No analyses yet.'}
          </p>
          {statusFilter ? (
            <p className="text-sm text-ink-400">Try a different filter.</p>
          ) : (
            <Link to="/" className="text-sm text-ink-700 underline hover:text-ink-900">
              Start your first analysis
            </Link>
          )}
        </div>
      ) : (
        <>
          {/* Journal table */}
          <table className="journal-table" role="table" aria-label="Analysis jobs">
            <thead>
              <tr>
                <th scope="col">Status</th>
                <th scope="col">Job ID</th>
                <th scope="col">Dataset</th>
                <th scope="col">Created</th>
                <th scope="col">Actions</th>
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
            <div className="flex items-center justify-end gap-4 mt-4 text-sm text-ink-600">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="hover:text-ink-900 disabled:text-ink-300 disabled:cursor-not-allowed"
                aria-label="Previous page"
              >
                Previous
              </button>
              <span aria-live="polite">
                Page {page + 1} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="hover:text-ink-900 disabled:text-ink-300 disabled:cursor-not-allowed"
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

  const { indicator, label, badgeClass } = statusDisplay(job.status);
  const datasetName = extractDatasetName(job.kaggle_url);

  return (
    <>
      <tr className="group">
        {/* Status */}
        <td>
          <span
            className={`inline-flex items-center px-2 py-0.5 text-xs font-medium ${badgeClass}`}
            style={{ borderRadius: 0 }}
          >
            {indicator}
            {label}
          </span>
        </td>

        {/* Job ID — monospace, truncated, links to detail */}
        <td>
          <Link
            to={`/jobs/${job.id}`}
            className="font-mono text-xs text-ink-700 hover:text-ink-900 hover:underline"
            title={job.id}
          >
            {job.id.length > 8 ? job.id.slice(0, 8) : job.id}
          </Link>
        </td>

        {/* Dataset — extracted name, links to Kaggle */}
        <td>
          <a
            href={job.kaggle_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-ink-700 hover:text-ink-900 hover:underline"
            title={job.kaggle_url}
          >
            {datasetName}
          </a>
        </td>

        {/* Created — relative time */}
        <td className="text-xs text-ink-500">
          {relativeTime(job.created_at)}
        </td>

        {/* Actions — small text buttons */}
        <td>
          <div className="flex items-center gap-3">
            {isRunning && (
              <button
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending}
                className="text-xs text-ink-600 hover:text-ink-900 disabled:text-ink-300"
                aria-label={`Stop job ${job.id}`}
              >
                {cancelMutation.isPending ? 'Stopping...' : 'Stop'}
              </button>
            )}
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="text-xs text-red-600 hover:text-red-800"
              aria-label={`Delete job ${job.id}`}
            >
              Delete
            </button>
          </div>
          {error && (
            <p className="text-xs text-red-600 mt-0.5">
              {error}{' '}
              <button onClick={() => setError(null)} className="underline">dismiss</button>
            </p>
          )}
        </td>
      </tr>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <tr>
          <td colSpan={5} className="!py-0">
            <div
              className="fixed inset-0 bg-black/40 flex items-center justify-center z-50"
              role="dialog"
              aria-modal="true"
              aria-labelledby={`delete-title-${job.id}`}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setShowDeleteConfirm(false);
              }}
            >
              <div className="bg-white border border-ink-200 p-6 max-w-md w-full mx-4">
                <h3 id={`delete-title-${job.id}`} className="font-serif text-lg font-bold text-ink-900 mb-2">
                  Delete Job {job.id.slice(0, 8)}?
                </h3>
                <p className="text-sm text-ink-600 mb-4">
                  This will permanently delete the job record, analysis results, and all associated data.
                  {isRunning && ' The job will be cancelled first.'}
                </p>
                <div className="flex justify-end gap-3">
                  <button
                    onClick={() => setShowDeleteConfirm(false)}
                    className="text-sm text-ink-600 hover:text-ink-900"
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
                    className="text-sm text-red-600 hover:text-red-800 font-medium disabled:text-red-300"
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
