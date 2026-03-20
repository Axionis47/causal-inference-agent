import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  CheckCircle,
  XCircle,
  Clock,
  Download,
  AlertTriangle,
  Activity,
  StopCircle,
  Trash2,
  Ban,
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
        <div className="w-8 h-8 border-4 border-gray-900 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (jobQuery.isError || !jobQuery.data) {
    return (
      <div className="card max-w-2xl mx-auto text-center">
        <XCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Job Not Found</h2>
        <p className="text-gray-600">
          The job you're looking for doesn't exist or has been deleted.
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
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-extrabold text-gray-900 tracking-tight mb-2">
              Analysis Job: {job.id}
            </h1>
            <p className="text-gray-500 break-all">{job.kaggle_url}</p>
          </div>
          <StatusBadge status={job.status} />
        </div>

        <div className="mt-6 grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Treatment Variable:</span>
            <p className="font-medium">{job.treatment_variable || 'Auto-detected'}</p>
          </div>
          <div>
            <span className="text-gray-500">Outcome Variable:</span>
            <p className="font-medium">{job.outcome_variable || 'Auto-detected'}</p>
          </div>
          <div>
            <span className="text-gray-500">Iterations:</span>
            <p className="font-medium">{job.iteration_count}</p>
          </div>
        </div>

        {isComplete && (
          <div className="mt-6">
            {resultsQuery.data?.notebook_url ? (
              <a
                href={getNotebookUrl(job.id)}
                download
                className="btn-primary inline-flex items-center space-x-2"
              >
                <Download className="w-5 h-5" />
                <span>Download Jupyter Notebook</span>
              </a>
            ) : resultsQuery.isSuccess && !resultsQuery.data?.notebook_url && (
              <p className="text-sm text-gray-500 italic">
                Notebook not available for this analysis.
              </p>
            )}
          </div>
        )}

        {isFailed && job.error_message && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4 flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Analysis Failed</p>
              <p className="text-sm text-red-600 mt-1">{job.error_message}</p>
            </div>
          </div>
        )}

        {isCancelled && (
          <div className="mt-6 bg-yellow-50 border border-yellow-200 rounded-xl p-4 flex items-start space-x-3">
            <Ban className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-yellow-800">Job Cancelled</p>
              <p className="text-sm text-yellow-600 mt-1">{job.error_message || 'This job was cancelled by user request.'}</p>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-6 flex items-center space-x-3">
          {isRunning && (
            <button
              onClick={() => cancelMutation.mutate()}
              disabled={cancelMutation.isPending}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-xl shadow-sm hover:shadow-md hover:bg-gray-200 disabled:opacity-50 transition-all"
            >
              <StopCircle className="w-5 h-5" />
              <span>{cancelMutation.isPending ? 'Cancelling...' : 'Stop Job'}</span>
            </button>
          )}

          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="inline-flex items-center space-x-2 px-4 py-2 bg-red-50 text-red-700 rounded-xl shadow-sm hover:shadow-md hover:bg-red-100 transition-all"
          >
            <Trash2 className="w-5 h-5" />
            <span>Delete Job</span>
          </button>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-job-title"
          onKeyDown={(e) => {
            if (e.key === 'Escape') setShowDeleteConfirm(false);
          }}
        >
          <div className="bg-white rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl">
            <h3 id="delete-job-title" className="text-lg font-bold text-gray-900 mb-2">Delete Job?</h3>
            <p className="text-gray-600 mb-4">
              This will permanently delete the job record, analysis results, and all associated data.
              {isRunning && ' The job will be cancelled first.'}
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-xl hover:bg-gray-200 transition-all"
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
                className="px-4 py-2 bg-red-600 text-white rounded-xl hover:bg-red-700 disabled:opacity-50 transition-all shadow-md"
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Progress */}
      {isRunning && (
        <>
          <JobProgress
            status={job.status}
            progress={job.progress_percentage ?? 0}
            currentAgent={job.current_agent}
          />
          <ActivityFeed events={agentEvents} />
        </>
      )}

      {/* Results */}
      {isComplete && (
        <>
          {resultsQuery.isLoading && (
            <div className="card">
              <div className="flex items-center justify-center py-12">
                <div className="w-8 h-8 border-4 border-gray-900 border-t-transparent rounded-full animate-spin" />
                <span className="ml-3 text-gray-600">Loading analysis results...</span>
              </div>
            </div>
          )}
          {resultsQuery.isError && (
            <div className="card">
              <div className="flex items-center space-x-3 text-red-600">
                <AlertTriangle className="w-6 h-6" />
                <div>
                  <p className="font-medium">Failed to load results</p>
                  <p className="text-sm text-red-500">Please try refreshing the page or check back later.</p>
                </div>
              </div>
            </div>
          )}
          {resultsQuery.data && (
            <ResultsDisplay results={resultsQuery.data} />
          )}
        </>
      )}

      {/* Agent Traces */}
      {(isComplete || isFailed) && <AgentTraces jobId={job.id} />}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { icon: React.ReactNode; class: string; label: string }> = {
    pending: {
      icon: <Clock className="w-3.5 h-3.5" />,
      class: 'bg-gray-100 text-gray-600 border border-gray-200',
      label: 'Pending',
    },
    completed: {
      icon: <CheckCircle className="w-3.5 h-3.5" />,
      class: 'bg-green-50 text-green-700 border border-green-200',
      label: 'Completed',
    },
    failed: {
      icon: <XCircle className="w-3.5 h-3.5" />,
      class: 'bg-red-50 text-red-700 border border-red-200',
      label: 'Failed',
    },
    cancelled: {
      icon: <Ban className="w-3.5 h-3.5" />,
      class: 'bg-yellow-50 text-yellow-700 border border-yellow-200',
      label: 'Cancelled',
    },
    cancelling: {
      icon: <StopCircle className="w-3.5 h-3.5" />,
      class: 'bg-yellow-50 text-yellow-700 border border-yellow-200',
      label: 'Cancelling...',
    },
  };

  const defaultConfig = {
    icon: <Activity className="w-3.5 h-3.5" />,
    class: 'bg-gray-900 text-white shadow-md',
    label: status.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
  };

  const { icon, class: className, label } = config[status] || defaultConfig;

  return (
    <span className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-semibold shadow-sm ${className}`}>
      {icon}
      <span>{label}</span>
    </span>
  );
}
