import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Clock, CheckCircle, XCircle, Activity, ChevronRight, Trash2, StopCircle, Ban } from 'lucide-react';
import { listJobs, Job, cancelJob, deleteJob } from '../services/api';

export default function JobsListPage() {
  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: () => listJobs(undefined, 50, 0),
  });

  if (jobsQuery.isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (jobsQuery.isError) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="card text-center py-12">
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Failed to load jobs</h3>
          <p className="text-gray-600 mb-4">
            {jobsQuery.error instanceof Error ? jobsQuery.error.message : 'An unexpected error occurred'}
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

  const jobs = jobsQuery.data?.jobs || [];

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Analysis Jobs</h1>
        <Link to="/" className="btn-primary">
          New Analysis
        </Link>
      </div>

      {jobs.length === 0 ? (
        <div className="card text-center py-12">
          <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No jobs yet</h3>
          <p className="text-gray-600 mb-4">
            Start your first causal analysis by submitting a Kaggle dataset.
          </p>
          <Link to="/" className="btn-primary">
            Start Analysis
          </Link>
        </div>
      ) : (
        <div className="space-y-4">
          {jobs.map((job) => (
            <JobCard key={job.id} job={job} />
          ))}
        </div>
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
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
    onError: (err: Error) => {
      setError(`Failed to delete: ${err.message}`);
    },
  });

  const statusConfig: Record<string, { icon: React.ReactNode; color: string }> = {
    completed: {
      icon: <CheckCircle className="w-5 h-5" />,
      color: 'text-green-500',
    },
    failed: {
      icon: <XCircle className="w-5 h-5" />,
      color: 'text-red-500',
    },
    pending: {
      icon: <Clock className="w-5 h-5" />,
      color: 'text-gray-400',
    },
    cancelled: {
      icon: <Ban className="w-5 h-5" />,
      color: 'text-yellow-500',
    },
  };

  const defaultStatus = {
    icon: <Activity className="w-5 h-5" />,
    color: 'text-blue-500',
  };

  const { icon, color } = statusConfig[job.status] || defaultStatus;
  const isRunning = !['completed', 'failed', 'pending', 'cancelled'].includes(job.status);

  return (
    <>
      <div className="card hover:shadow-md transition-shadow group">
        {error && (
          <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between">
            <span className="text-sm text-red-700">{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
            >
              <XCircle className="w-4 h-4" />
            </button>
          </div>
        )}
        <div className="flex items-center justify-between">
          <Link to={`/jobs/${job.id}`} className="flex items-center space-x-4 flex-1">
            <div className={color}>
              {isRunning ? (
                <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
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
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>

            <Link to={`/jobs/${job.id}`}>
              <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-gray-600" />
            </Link>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Job {job.id}?</h3>
            <p className="text-gray-600 mb-4">
              This will permanently delete the job record, analysis results, and all associated data.
              {isRunning && ' The job will be cancelled first.'}
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
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
