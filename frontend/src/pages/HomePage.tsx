import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { AlertCircle, ChevronDown, ChevronRight } from 'lucide-react';
import { createJob, CreateJobRequest, OrchestratorMode } from '../services/api';
import { validateKaggleUrl } from '../utils';

const ORCHESTRATOR_CAPTIONS: Record<OrchestratorMode, string> = {
  standard: 'Fixed pipeline of 13 agents.',
  react: 'Orchestrator decides which agent runs next (experimental).',
};

export default function HomePage() {
  const navigate = useNavigate();
  const [kaggleUrl, setKaggleUrl] = useState('');
  const [treatmentVar, setTreatmentVar] = useState('');
  const [outcomeVar, setOutcomeVar] = useState('');
  const [causalQuestion, setCausalQuestion] = useState('');
  const [orchestratorMode, setOrchestratorMode] = useState<OrchestratorMode>('standard');
  const [showAdvanced, setShowAdvanced] = useState(false);
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
    if (!treatmentVar.trim()) { setError('Treatment variable is required'); return; }
    if (!outcomeVar.trim()) { setError('Outcome variable is required'); return; }
    createJobMutation.mutate({
      kaggle_url: kaggleUrl.trim(),
      treatment_variable: treatmentVar.trim(),
      outcome_variable: outcomeVar.trim(),
      orchestrator_mode: orchestratorMode,
      user_context: causalQuestion.trim() || undefined,
    });
  };

  const inputCls =
    'w-full px-3 py-2 font-mono text-sm text-ink bg-canvas-inset border border-edge-subtle ' +
    'placeholder:text-ink-tertiary focus:border-edge-strong transition-colors';
  const labelCls = 'block text-2xs font-mono uppercase tracking-[0.15em] text-ink-tertiary mb-1.5';

  return (
    <div className="max-w-2xl mx-auto">
      {/* Title block */}
      <div className="border-b border-edge pb-6 mb-10">
        <h1 className="text-xl font-semibold text-ink mb-3">
          Automated Causal Inference Analysis
        </h1>
        <p className="text-ink-secondary text-sm leading-relaxed max-w-lg">
          Submit a Kaggle dataset to run a multi-agent pipeline: data profiling,
          causal discovery, treatment effect estimation, sensitivity analysis,
          and reproducible notebook generation.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="kaggle-url" className={labelCls}>
            Dataset URL
          </label>
          <input
            id="kaggle-url"
            type="text"
            value={kaggleUrl}
            onChange={(e) => setKaggleUrl(e.target.value)}
            placeholder="https://www.kaggle.com/datasets/owner/dataset-name"
            className={inputCls}
          />
          <p className="mt-1.5 text-xs text-ink-tertiary">
            Full URL to a public Kaggle dataset
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label htmlFor="treatment-var" className={labelCls}>
              Treatment variable
            </label>
            <input
              id="treatment-var"
              type="text"
              value={treatmentVar}
              onChange={(e) => setTreatmentVar(e.target.value)}
              placeholder="e.g. treat"
              className={inputCls}
            />
          </div>
          <div>
            <label htmlFor="outcome-var" className={labelCls}>
              Outcome variable
            </label>
            <input
              id="outcome-var"
              type="text"
              value={outcomeVar}
              onChange={(e) => setOutcomeVar(e.target.value)}
              placeholder="e.g. re78"
              className={inputCls}
            />
          </div>
        </div>

        <div>
          <label htmlFor="causal-question" className={labelCls}>
            Context <span className="text-ink-tertiary normal-case">(optional)</span>
          </label>
          <textarea
            id="causal-question"
            value={causalQuestion}
            onChange={(e) => setCausalQuestion(e.target.value)}
            placeholder="What the variables mean, the study it came from, known confounders, whether it was a randomized trial."
            rows={3}
            className={inputCls + ' resize-y'}
          />
          <p className="mt-1.5 text-xs text-ink-tertiary">
            Helps the domain-knowledge agent when the dataset has no description
          </p>
        </div>

        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex items-center gap-1 text-xs text-ink-secondary hover:text-ink"
            aria-expanded={showAdvanced}
          >
            {showAdvanced ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            Advanced options
          </button>
          {showAdvanced && (
            <div className="mt-3 pl-4 border-l border-edge-subtle">
              <label className={labelCls}>Orchestrator</label>
              <div role="group" aria-label="Orchestrator" className="inline-flex">
                {(['standard', 'react'] as const).map((mode, i) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setOrchestratorMode(mode)}
                    aria-pressed={orchestratorMode === mode}
                    className={[
                      'px-3 py-1.5 text-xs font-mono border border-edge-subtle',
                      i === 0 ? 'rounded-l-md' : 'rounded-r-md border-l-0',
                      orchestratorMode === mode
                        ? 'bg-indigo/15 text-indigo border-indigo'
                        : 'bg-transparent text-ink-secondary hover:text-ink',
                    ].join(' ')}
                  >
                    {mode === 'standard' ? 'Standard' : 'ReAct'}
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-xs text-ink-tertiary">
                {ORCHESTRATOR_CAPTIONS[orchestratorMode]}
              </p>
            </div>
          )}
        </div>

        {error && (
          <div
            className="flex items-start gap-2 text-rose bg-rose/10 border border-rose/30 rounded-md p-3 text-sm"
            role="alert"
          >
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <button
          type="submit"
          disabled={createJobMutation.isPending}
          className="w-full bg-mint text-ink-inverse rounded-md py-2.5 text-sm font-medium hover:bg-mint-bright disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {createJobMutation.isPending ? 'Starting Analysis...' : 'Run Causal Analysis'}
        </button>
      </form>

      {/* Method summary */}
      <div className="mt-12 pt-6 border-t border-edge-subtle">
        <p className="text-xs text-ink-tertiary leading-relaxed">
          <span className="font-medium text-ink-secondary">Pipeline:</span>{' '}
          13 specialist agents coordinated by an LLM orchestrator.
          12 estimation methods (OLS, IPW, AIPW, PSM, DiD, IV, RDD, S/T/X-Learner, Causal Forest, Double ML).
          5 discovery algorithms (PC, FCI, GES, NOTEARS, LiNGAM).
          Sensitivity analysis via E-value, Rosenbaum bounds, placebo tests, and specification curves.
        </p>
      </div>
    </div>
  );
}
