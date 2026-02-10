import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Clock, CheckCircle, XCircle, Activity, ChevronRight, Trash2, StopCircle, Ban, ChevronLeft, Filter } from 'lucide-react';
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

  // Loading skeleton
  if (jobsQuery.isLoading) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="h-8 w-48 bg-gray-200 rounded animate-pulse" />
          <div className="h-10 w-32 bg-gray-200 rounded animate-pulse" />
        </div>
        <div className="space-y-4" aria-label="Loading jobs">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card animate-pulse">
              <div className="flex items-center space-x-4">
                <div className="w-5 h-5 bg-gray-200 rounded-full" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 w-32 bg-gray-200 rounded" />
                  <div className="h-3 w-64 bg-gray-200 rounded" />
                </div>
                <div className="h-4 w-20 bg-gray-200 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (jobsQuery.isError) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="card text-center py-12" role="alert">
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" aria-hidden="true" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Failed to load jobs</h3>
          <p className="text-gray-600 mb-4">
            {(jobsQuery.error as { message?: string })?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={() => jobsQuery.refetch()}
            className="btn-primary"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Analysis Jobs</h1>
        <Link to="/" className="btn-primary">
          New Analysis
        </Link>
      </div>

      {/* Filters */}
      <div className="flex items-center space-x-3 mb-4">
        <Filter className="w-4 h-4 text-gray-500" aria-hidden="true" />
        <label htmlFor="status-filter" className="sr-only">Filter by status</label>
        <select
          id="status-filter"
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPage(0); // Reset to first page on filter change
          }}
          className="border border-gray-300 rounded-lg px-3 py-1.5 text-sm bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          aria-label="Filter jobs by status"
        >
          {STATUS_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        {totalJobs > 0 && (
          <span className="text-sm text-gray-500">
            {totalJobs} job{totalJobs !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {jobs.length === 0 ? (
        <div className="card text-center py-12">
          <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" aria-hidden="true" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {statusFilter ? 'No matching jobs' : 'No jobs yet'}
          </h3>
          <p className="text-gray-600 mb-4">
            {statusFilter
              ? `No jobs found with status "${statusFilter}". Try a different filter.`
              : 'Start your first causal analysis by submitting a Kaggle dataset.'
            }
          </p>
          {!statusFilter && (
            <Link to="/" className="btn-primary">
              Start Analysis
            </Link>
          )}
        </div>
      ) : (
        <>
          <div className="space-y-4" role="list" aria-label="Analysis jobs">
            {jobs.map((job) => (
              <div key={job.id} role="listitem">
                <JobCard job={job} />
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-6">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="inline-flex items-center space-x-1 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Previous page"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Previous</span>
              </button>
              <span className="text-sm text-gray-600" aria-live="polite">
                Page {page + 1} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="inline-flex items-center space-x-1 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Next page"
              >
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function JobCard({ job }: { job: Job }) {
  const queryClient = useQueryClient();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const statusConfig: Record<string, { icon: React.ReactNode; color: string }> = {
    completed: {
      icon: <CheckCircle className="w-5 h-5" aria-hidden="true" />,
      color: 'text-green-500',
    },
    failed: {
      icon: <XCircle className="w-5 h-5" aria-hidden="true" />,
      color: 'text-red-500',
    },
    pending: {
      icon: <Clock className="w-5 h-5" aria-hidden="true" />,
      color: 'text-gray-400',
    },
    cancelled: {
      icon: <Ban className="w-5 h-5" aria-hidden="true" />,
      color: 'text-yellow-500',
    },
  };

  const defaultStatus = {
    icon: <Activity className="w-5 h-5" aria-hidden="true" />,
    color: 'text-blue-500',
  };

  const { icon, color } = statusConfig[job.status] || defaultStatus;
  const isRunning = !['completed', 'failed', 'pending', 'cancelled'].includes(job.status);

  return (
    <>
      <div className="card hover:shadow-md transition-shadow group">
        {error && (
          <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between" role="alert">
            <span className="text-sm text-red-700">{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
              aria-label="Dismiss error"
            >
              <XCircle className="w-4 h-4" />
            </button>
          </div>
        )}
        <div className="flex items-center justify-between">
          <Link to={`/jobs/${job.id}`} className="flex items-center space-x-4 flex-1">
            <div className={color}>
              {isRunning ? (
                <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" role="status" aria-label="Running" />
              ) : (
                icon
              )}
            </div>
            <div>
              <p className="font-medium text-gray-900 group-hover:text-primary-600">
                Job {job.id}
              </p>
              <p className="text-sm text-gray-500 truncate max-w-md">
                {job.kaggle_url}
              </p>
            </div>
          </Link>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900 capitalize">
                {job.status.replace('_', ' ')}
              </p>
              <p className="text-xs text-gray-500">
                {new Date(job.created_at).toLocaleDateString()}
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center space-x-2">
              {isRunning && (
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    cancelMutation.mutate();
                  }}
                  disabled={cancelMutation.isPending}
                  className="p-2 text-yellow-600 hover:bg-yellow-50 rounded-lg transition-colors"
                  title="Stop Job"
                  aria-label={`Stop job ${job.id}`}
                >
                  <StopCircle className="w-5 h-5" />
                </button>
              )}
              <button
                onClick={(e) => {
                  e.preventDefault();
                  setShowDeleteConfirm(true);
                }}
                className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                title="Delete Job"
                aria-label={`Delete job ${job.id}`}
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>

            <Link to={`/jobs/${job.id}`} aria-label={`View job ${job.id} details`}>
              <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-gray-600" />
            </Link>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          role="dialog"
          aria-modal="true"
          aria-labelledby={`delete-title-${job.id}`}
          onKeyDown={(e) => {
            if (e.key === 'Escape') setShowDeleteConfirm(false);
          }}
        >
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 id={`delete-title-${job.id}`} className="text-lg font-semibold text-gray-900 mb-2">
              Delete Job {job.id}?
            </h3>
            <p className="text-gray-600 mb-4">
              This will permanently delete the job record, analysis results, and all associated data.
              {isRunning && ' The job will be cancelled first.'}
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
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
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
