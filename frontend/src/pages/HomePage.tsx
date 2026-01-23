import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { Upload, AlertCircle, ArrowRight } from 'lucide-react';
import { createJob, CreateJobRequest } from '../services/api';

export default function HomePage() {
  const navigate = useNavigate();
  const [kaggleUrl, setKaggleUrl] = useState('');
  const [treatmentVar, setTreatmentVar] = useState('');
  const [outcomeVar, setOutcomeVar] = useState('');
  const [error, setError] = useState<string | null>(null);

  const createJobMutation = useMutation({
    mutationFn: (request: CreateJobRequest) => createJob(request),
    onSuccess: (job) => {
      navigate(`/jobs/${job.id}`);
    },
    onError: (err: Error) => {
      setError(err.message || 'Failed to create job');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!kaggleUrl.trim()) {
      setError('Please enter a Kaggle dataset URL');
      return;
    }

    createJobMutation.mutate({
      kaggle_url: kaggleUrl.trim(),
      treatment_variable: treatmentVar.trim() || undefined,
      outcome_variable: outcomeVar.trim() || undefined,
    });
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Causal Inference Analysis
        </h1>
        <p className="text-lg text-gray-600">
          Enter a Kaggle dataset URL to automatically analyze causal relationships
          using AI-powered agents.
        </p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label
              htmlFor="kaggle-url"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Kaggle Dataset URL *
            </label>
            <input
              id="kaggle-url"
              type="text"
              value={kaggleUrl}
              onChange={(e) => setKaggleUrl(e.target.value)}
              placeholder="https://www.kaggle.com/datasets/username/dataset-name"
              className="input-field"
            />
            <p className="mt-1 text-sm text-gray-500">
              Paste the full URL of a Kaggle dataset
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="treatment-var"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Treatment Variable (optional)
              </label>
              <input
                id="treatment-var"
                type="text"
                value={treatmentVar}
                onChange={(e) => setTreatmentVar(e.target.value)}
                placeholder="e.g., treatment, intervention"
                className="input-field"
              />
            </div>

            <div>
              <label
                htmlFor="outcome-var"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Outcome Variable (optional)
              </label>
              <input
                id="outcome-var"
                type="text"
                value={outcomeVar}
                onChange={(e) => setOutcomeVar(e.target.value)}
                placeholder="e.g., outcome, response"
                className="input-field"
              />
            </div>
          </div>

          {error && (
            <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-lg">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={createJobMutation.isPending}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            {createJobMutation.isPending ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Starting Analysis...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Start Causal Analysis</span>
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </form>
      </div>

      <div className="mt-12 grid grid-cols-3 gap-6">
        <div className="text-center">
          <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-2xl">1</span>
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">Data Profiling</h3>
          <p className="text-sm text-gray-600">
            AI agents analyze your dataset to identify treatment and outcome variables
          </p>
        </div>

        <div className="text-center">
          <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-2xl">2</span>
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">Effect Estimation</h3>
          <p className="text-sm text-gray-600">
            Multiple causal inference methods estimate treatment effects
          </p>
        </div>

        <div className="text-center">
          <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-2xl">3</span>
          </div>
          <h3 className="font-semibold text-gray-900 mb-2">Reproducible Report</h3>
          <p className="text-sm text-gray-600">
            Download a Jupyter notebook with all analysis code and results
          </p>
        </div>
      </div>
    </div>
  );
}
