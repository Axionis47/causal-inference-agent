// The analysis panel for the new analysis slice: a compact stage-progress
// strip, one panel per agent (each owning its artifact lines), and the run
// cost line. Presentational: takes the full AnalysisViewResponse; JobPage
// owns fetching (useAnalysis) and live updates (SSE invalidation in useJob).

import type {
  AnalysisArtifactView,
  AnalysisRunStatus,
  AnalysisViewResponse,
} from '../../../services/api';
import { Caption } from '../terminal/atoms';
import { ArtifactLine } from './artifacts/ArtifactLine';
import { CostLine } from './CostLine';
import { NotebookDownloadCard } from './NotebookDownloadCard';
import { AgentPanel } from './panels/AgentPanel';
import { PlanGate } from './PlanGate';

const noop = () => {};

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';

const RUN_DOT: Record<AnalysisRunStatus, string> = {
  pending: 'bg-edge-subtle',
  running: 'bg-amber animate-pulse-live',
  waiting_for_user: 'bg-amber',
  failed: 'bg-rose',
  completed: 'bg-mint',
  cancelled: 'bg-ink-tertiary',
};

const RUN_TEXT: Record<AnalysisRunStatus, string> = {
  pending: 'text-ink-tertiary',
  running: 'text-amber',
  waiting_for_user: 'text-amber',
  failed: 'text-rose',
  completed: 'text-mint',
  cancelled: 'text-ink-tertiary',
};

/** Compact 13-cell strip: done stages mint, the current one amber, rest dim. */
function StageStrip({
  stageIndex,
  totalStages,
  status,
}: {
  stageIndex: number;
  totalStages: number;
  status: AnalysisRunStatus;
}) {
  const cells = Array.from({ length: totalStages }, (_, i) => {
    let cls = 'bg-edge-subtle';
    if (status === 'completed' || i < stageIndex) cls = 'bg-mint';
    else if (i === stageIndex) {
      cls = status === 'running' ? 'bg-amber animate-pulse-live' : 'bg-amber';
      if (status === 'failed') cls = 'bg-rose';
    }
    return <span key={i} className={`h-1 flex-1 ${cls}`} />;
  });
  return (
    <div className="flex items-center gap-px" aria-label="analysis stage progress">
      {cells}
    </div>
  );
}

export function AnalysisView({
  analysis,
  onPlanConfirm,
  onPlanReject,
  planSubmitting = false,
  planError = null,
}: {
  analysis: AnalysisViewResponse;
  // Plan-gate handlers; JobPage owns the POST /jobs/:id/plan mutation.
  onPlanConfirm?: (edits: Record<string, string>) => void;
  onPlanReject?: (reason: string) => void;
  planSubmitting?: boolean;
  planError?: string | null;
}) {
  const { spec_summary: spec, plan_gate: planGate, method_plan: methodPlan } = analysis;
  const showPlanGate =
    analysis.status === 'waiting_for_user' && !!planGate?.confirmation_card;

  // Each artifact renders inside the panel of the agent that emitted it.
  // Anything whose agent has no run entry falls through to a trailing list
  // so no artifact silently disappears.
  const byAgent = new Map<string, AnalysisArtifactView[]>();
  for (const a of analysis.artifacts) {
    const list = byAgent.get(a.agent);
    if (list) list.push(a);
    else byAgent.set(a.agent, [a]);
  }
  const knownAgents = new Set(analysis.agents.map((a) => a.agent));
  const orphanArtifacts = analysis.artifacts.filter((a) => !knownAgents.has(a.agent));

  return (
    <section className="flex-1 min-w-0 flex flex-col overflow-hidden bg-canvas">
      <Caption>[ analysis ]</Caption>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* run header: status, question, stage strip, spec summary, error */}
        <div className="bg-canvas-raised border border-edge-subtle rounded-md p-3 space-y-2">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${RUN_DOT[analysis.status]}`}
            />
            <span className={`${labelCls} ${RUN_TEXT[analysis.status]}`}>
              {analysis.status.replace(/_/g, ' ')}
            </span>
            <span className="ml-auto font-mono text-2xs text-ink-tertiary tabular shrink-0">
              stage {analysis.stage_index}/{analysis.total_stages} ·{' '}
              {analysis.current_state}
            </span>
          </div>

          <StageStrip
            stageIndex={analysis.stage_index}
            totalStages={analysis.total_stages}
            status={analysis.status}
          />

          <p className="text-sm text-ink">{analysis.causal_question}</p>

          {spec && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-2xs text-ink-tertiary">
              <span>
                type <span className="text-ink-secondary">{spec.question_type}</span>
              </span>
              <span>
                confidence <span className="text-ink-secondary">{spec.confidence}</span>
              </span>
              <span>
                treatment{' '}
                <span className="text-ink-secondary">{spec.treatment ?? '—'}</span>
              </span>
              <span>
                outcome <span className="text-ink-secondary">{spec.outcome ?? '—'}</span>
              </span>
            </div>
          )}

          {/* compact method-plan line, whenever planning has produced one */}
          {methodPlan && (
            <div className="font-mono text-2xs text-ink-tertiary">
              plan{' '}
              <span className="text-ink-secondary">
                {[
                  methodPlan.lane,
                  methodPlan.estimator,
                  methodPlan.estimand,
                  methodPlan.outcome,
                ].join(' · ')}
              </span>
            </div>
          )}

          {planGate?.status === 'pass_auto_approved' && (
            <p className="text-2xs font-mono text-mint">plan auto-approved</p>
          )}

          {analysis.error_message && (
            <p className="text-xs font-mono text-rose">{analysis.error_message}</p>
          )}
        </div>

        {/* the deliverable: notebook download once verification recorded a result */}
        {analysis.notebook && (
          <NotebookDownloadCard jobId={analysis.job_id} notebook={analysis.notebook} />
        )}

        {/* the plan gate, prominent above the agent tiles while the run waits */}
        {showPlanGate && planGate && (
          <PlanGate
            gate={planGate}
            design={analysis.selected_design}
            methodPlan={methodPlan}
            onConfirm={onPlanConfirm ?? noop}
            onReject={onPlanReject ?? noop}
            submitting={planSubmitting}
            error={planError}
          />
        )}

        {/* one panel per agent, in spine order, each owning its artifacts */}
        <div>
          <div className={`${labelCls} text-ink-tertiary mb-1.5`}>[ agents ]</div>
          <div className="space-y-2">
            {analysis.agents.map((a) => (
              <AgentPanel
                key={a.agent}
                jobId={analysis.job_id}
                agent={a}
                artifacts={byAgent.get(a.agent) ?? []}
              />
            ))}
          </div>
        </div>

        {/* artifacts from agents without a run entry; normally empty */}
        {orphanArtifacts.length > 0 && (
          <div>
            <div className={`${labelCls} text-ink-tertiary mb-1.5`}>
              [ other artifacts ]
            </div>
            {orphanArtifacts.map((a) => (
              <ArtifactLine key={a.artifact_id} jobId={analysis.job_id} artifact={a} />
            ))}
          </div>
        )}

        {/* run totals */}
        <div className="border-t border-edge-subtle pt-2">
          <CostLine costs={analysis.costs} />
        </div>
      </div>
    </section>
  );
}
