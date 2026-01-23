import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  CheckCircle,
  XCircle,
  Clock,
  Download,
  AlertTriangle,
  Activity,
} from 'lucide-react';
import { getJob, getJobStatus, getResults, getNotebookUrl } from '../services/api';
import JobProgress from '../components/job/JobProgress';
import ResultsDisplay from '../components/results/ResultsDisplay';
import AgentTraces from '../components/agents/AgentTraces';

export default function JobPage() {
  const { jobId } = useParams<{ jobId: string }>();

  const jobQuery = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
  });

  const statusQuery = useQuery({
    queryKey: ['jobStatus', jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: !!jobId && jobQuery.data?.status !== 'completed' && jobQuery.data?.status !== 'failed',
    refetchInterval: 3000, // Poll every 3 seconds while running
  });

  const resultsQuery = useQuery({
    queryKey: ['results', jobId],
    queryFn: () => getResults(jobId!),
    enabled: jobQuery.data?.status === 'completed',
  });

  if (jobQuery.isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
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
  const status = statusQuery.data || { progress_percentage: 0, current_agent: undefined };
  const isComplete = job.status === 'completed';
  const isFailed = job.status === 'failed';
  const isRunning = !isComplete && !isFailed;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              Analysis Job: {job.id}
            </h1>
            <p className="text-gray-600 break-all">{job.kaggle_url}</p>
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

        {isComplete && resultsQuery.data?.notebook_url && (
          <div className="mt-6">
            <a
              href={getNotebookUrl(job.id)}
              download
              className="btn-primary inline-flex items-center space-x-2"
            >
              <Download className="w-5 h-5" />
              <span>Download Jupyter Notebook</span>
            </a>
          </div>
        )}

        {isFailed && job.error_message && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Analysis Failed</p>
              <p className="text-sm text-red-600 mt-1">{job.error_message}</p>
            </div>
          </div>
        )}
      </div>

      {/* Progress */}
      {isRunning && (
        <JobProgress
          status={job.status}
          progress={status.progress_percentage}
          currentAgent={job.current_agent || status.current_agent}
        />
      )}

      {/* Results */}
      {isComplete && resultsQuery.data && (
        <ResultsDisplay results={resultsQuery.data} />
      )}

      {/* Agent Traces */}
      {(isComplete || isFailed) && <AgentTraces jobId={job.id} />}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { icon: React.ReactNode; class: string; label: string }> = {
    pending: {
      icon: <Clock className="w-4 h-4" />,
      class: 'bg-gray-100 text-gray-700',
      label: 'Pending',
    },
    completed: {
      icon: <CheckCircle className="w-4 h-4" />,
      class: 'bg-green-100 text-green-700',
      label: 'Completed',
    },
    failed: {
      icon: <XCircle className="w-4 h-4" />,
      class: 'bg-red-100 text-red-700',
      label: 'Failed',
    },
  };

  const defaultConfig = {
    icon: <Activity className="w-4 h-4" />,
    class: 'bg-blue-100 text-blue-700',
    label: status.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
  };

  const { icon, class: className, label } = config[status] || defaultConfig;

  return (
    <span className={`inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-sm font-medium ${className}`}>
      {icon}
      <span>{label}</span>
    </span>
  );
}
