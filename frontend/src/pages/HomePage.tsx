import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { AlertCircle } from 'lucide-react';
import { createJob, CreateJobRequest } from '../services/api';
import { validateKaggleUrl } from '../utils';

export default function HomePage() {
  const navigate = useNavigate();
  const [kaggleUrl, setKaggleUrl] = useState('');
  const [treatmentVar, setTreatmentVar] = useState('');
  const [outcomeVar, setOutcomeVar] = useState('');
  const [error, setError] = useState<string | null>(null);

  const createJobMutation = useMutation({
    mutationFn: (request: CreateJobRequest) => createJob(request),
    onSuccess: (job) => navigate(`/jobs/${job.id}`),
    onError: (err: Error) => setError(err.message || 'Failed to create job'),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const validationError = validateKaggleUrl(kaggleUrl.trim());
    if (validationError) { setError(validationError); return; }
    createJobMutation.mutate({
      kaggle_url: kaggleUrl.trim(),
      treatment_variable: treatmentVar.trim() || undefined,
      outcome_variable: outcomeVar.trim() || undefined,
    });
  };

  return (
    <div className="max-w-2xl mx-auto">
      {/* Title block — journal style */}
      <div className="border-b-2 border-ink-900 pb-6 mb-10">
        <h1 className="font-serif text-3xl font-700 text-ink-900 mb-3">
          Automated Causal Inference Analysis
        </h1>
        <p className="text-ink-500 text-sm leading-relaxed max-w-lg">
          Submit a Kaggle dataset to run a multi-agent pipeline: data profiling,
          causal discovery, treatment effect estimation, sensitivity analysis,
          and reproducible notebook generation.
        </p>
      </div>

      {/* Form — clean, no card wrapper */}
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="kaggle-url" className="block text-sm font-medium text-ink-700 mb-1.5">
            Dataset URL
          </label>
          <input
            id="kaggle-url"
            type="text"
            value={kaggleUrl}
            onChange={(e) => setKaggleUrl(e.target.value)}
            placeholder="https://www.kaggle.com/datasets/owner/dataset-name"
            className="input-field font-mono text-sm"
          />
          <p className="mt-1.5 text-xs text-ink-300">
            Full URL to a public Kaggle dataset
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="treatment-var" className="block text-sm font-medium text-ink-700 mb-1.5">
              Treatment variable <span className="text-ink-300 font-normal">(optional)</span>
            </label>
            <input
              id="treatment-var"
              type="text"
              value={treatmentVar}
              onChange={(e) => setTreatmentVar(e.target.value)}
              placeholder="e.g. treatment"
              className="input-field font-mono text-sm"
            />
          </div>
          <div>
            <label htmlFor="outcome-var" className="block text-sm font-medium text-ink-700 mb-1.5">
              Outcome variable <span className="text-ink-300 font-normal">(optional)</span>
            </label>
            <input
              id="outcome-var"
              type="text"
              value={outcomeVar}
              onChange={(e) => setOutcomeVar(e.target.value)}
              placeholder="e.g. earnings"
              className="input-field font-mono text-sm"
            />
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-2 text-sig-no bg-red-50 border border-red-200 p-3 text-sm">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <button
          type="submit"
          disabled={createJobMutation.isPending}
          className="btn-primary w-full"
        >
          {createJobMutation.isPending ? 'Starting Analysis...' : 'Run Causal Analysis'}
        </button>
      </form>

      {/* Method summary — like a journal abstract footer */}
      <div className="mt-12 pt-6 border-t border-ink-100">
        <p className="text-xs text-ink-300 leading-relaxed">
          <span className="font-medium text-ink-500">Pipeline:</span>{' '}
          13 specialist agents coordinated by an LLM orchestrator.
          12 estimation methods (OLS, IPW, AIPW, PSM, DiD, IV, RDD, S/T/X-Learner, Causal Forest, Double ML).
          5 discovery algorithms (PC, FCI, GES, NOTEARS, LiNGAM).
          Sensitivity analysis via E-value, Rosenbaum bounds, placebo tests, and specification curves.
        </p>
      </div>
    </div>
  );
}
