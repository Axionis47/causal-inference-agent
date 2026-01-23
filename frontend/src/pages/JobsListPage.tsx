import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Clock, CheckCircle, XCircle, Activity, ChevronRight } from 'lucide-react';
import { listJobs, Job } from '../services/api';

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
  };

  const defaultStatus = {
    icon: <Activity className="w-5 h-5" />,
    color: 'text-blue-500',
  };

  const { icon, color } = statusConfig[job.status] || defaultStatus;
  const isRunning = !['completed', 'failed', 'pending'].includes(job.status);

  return (
    <Link
      to={`/jobs/${job.id}`}
      className="card block hover:shadow-md transition-shadow group"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
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
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-right">
            <p className="text-sm font-medium text-gray-900 capitalize">
              {job.status.replace('_', ' ')}
            </p>
            <p className="text-xs text-gray-500">
              {new Date(job.created_at).toLocaleDateString()}
            </p>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-gray-600" />
        </div>
      </div>
    </Link>
  );
}
