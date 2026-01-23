import { Activity, Check } from 'lucide-react';

interface JobProgressProps {
  status: string;
  progress: number;
  currentAgent?: string;
}

const STAGES = [
  { key: 'fetching_data', label: 'Fetching Data', agent: 'data_loader' },
  { key: 'profiling', label: 'Data Profiling', agent: 'data_profiler' },
  { key: 'discovering_causal', label: 'Causal Discovery', agent: 'causal_discovery' },
  { key: 'estimating_effects', label: 'Effect Estimation', agent: 'effect_estimator' },
  { key: 'sensitivity_analysis', label: 'Sensitivity Analysis', agent: 'sensitivity_analyst' },
  { key: 'critique_review', label: 'Quality Review', agent: 'critique' },
  { key: 'generating_notebook', label: 'Generating Notebook', agent: 'notebook_generator' },
];

export default function JobProgress({ status, progress, currentAgent }: JobProgressProps) {
  const currentStageIndex = STAGES.findIndex((s) => s.key === status);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900">Analysis Progress</h2>
        <span className="text-sm font-medium text-gray-600">{progress}%</span>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-gray-200 rounded-full h-2 mb-8">
        <div
          className="bg-primary-600 h-2 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Stages */}
      <div className="relative">
        {/* Connector line */}
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200" />

        <div className="space-y-6">
          {STAGES.map((stage, index) => {
            const isComplete = index < currentStageIndex;
            const isCurrent = index === currentStageIndex;

            return (
              <div key={stage.key} className="relative flex items-start pl-10">
                {/* Status indicator */}
                <div
                  className={`absolute left-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    isComplete
                      ? 'bg-green-100'
                      : isCurrent
                      ? 'bg-primary-100'
                      : 'bg-gray-100'
                  }`}
                >
                  {isComplete ? (
                    <Check className="w-4 h-4 text-green-600" />
                  ) : isCurrent ? (
                    <Activity className="w-4 h-4 text-primary-600 animate-pulse" />
                  ) : (
                    <div className="w-2 h-2 rounded-full bg-gray-300" />
                  )}
                </div>

                {/* Stage info */}
                <div>
                  <p
                    className={`font-medium ${
                      isComplete
                        ? 'text-green-700'
                        : isCurrent
                        ? 'text-primary-700'
                        : 'text-gray-400'
                    }`}
                  >
                    {stage.label}
                  </p>
                  {isCurrent && currentAgent && (
                    <p className="text-sm text-gray-500 mt-1">
                      Agent: {currentAgent.replace('_', ' ')}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
