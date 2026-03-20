import { Check } from 'lucide-react';

interface JobProgressProps {
  status: string;
  progress: number;
  currentAgent?: string;
}

const STAGES = [
  { key: 'fetching_data', label: 'Data Fetch' },
  { key: 'profiling', label: 'Profiling' },
  { key: 'exploratory_analysis', label: 'EDA' },
  { key: 'discovering_causal', label: 'Causal Discovery' },
  { key: 'estimating_effects', label: 'Effect Estimation' },
  { key: 'sensitivity_analysis', label: 'Sensitivity' },
  { key: 'critique_review', label: 'Quality Review' },
  { key: 'generating_notebook', label: 'Notebook' },
];

export default function JobProgress({ status, progress, currentAgent }: JobProgressProps) {
  const currentStageIndex = STAGES.findIndex((s) => s.key === status);

  return (
    <div className="section">
      <div className="flex items-baseline justify-between mb-6">
        <h2 className="font-serif text-lg font-600 text-ink-900">Analysis Progress</h2>
        <span className="font-mono text-sm text-ink-500">{progress}%</span>
      </div>

      {/* Thin progress bar */}
      <div className="w-full bg-ink-100 h-px mb-6 relative">
        <div
          className="bg-accent h-0.5 absolute top-0 left-0 transition-all duration-1000"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Vertical timeline */}
      <div className="space-y-0">
        {STAGES.map((stage, index) => {
          const isComplete = index < currentStageIndex;
          const isCurrent = index === currentStageIndex;

          return (
            <div key={stage.key} className="flex items-start gap-3 relative">
              {/* Vertical line */}
              {index < STAGES.length - 1 && (
                <div className={`absolute left-[7px] top-5 w-px h-full ${
                  isComplete ? 'bg-accent' : 'bg-ink-100'
                }`} />
              )}

              {/* Marker */}
              <div className="flex-shrink-0 relative z-10 mt-0.5">
                {isComplete ? (
                  <div className="w-4 h-4 bg-accent flex items-center justify-center">
                    <Check className="w-3 h-3 text-white" strokeWidth={3} />
                  </div>
                ) : isCurrent ? (
                  <div className="w-4 h-4 border-2 border-accent bg-white">
                    <div className="w-1.5 h-1.5 bg-accent m-[3px] animate-pulse" />
                  </div>
                ) : (
                  <div className="w-4 h-4 border border-ink-200 bg-white" />
                )}
              </div>

              {/* Label */}
              <div className={`pb-4 ${isCurrent ? 'text-ink-900' : isComplete ? 'text-ink-500' : 'text-ink-300'}`}>
                <span className={`text-sm ${isCurrent ? 'font-medium' : ''}`}>
                  {stage.label}
                </span>
                {isCurrent && currentAgent && (
                  <span className="ml-2 font-mono text-xs text-accent">
                    {currentAgent}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
