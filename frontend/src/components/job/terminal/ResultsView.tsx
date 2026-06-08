// Completed-job results overlay. There is no separate results route; the F5
// "results" key and a finished job open this over the terminal. It fetches the
// final analysis from /jobs/{id}/results and renders the executive summary, the
// effect forest plot, method consensus, sensitivity, and the notebook download.
// Without it, a completed job has nowhere to show its actual output.

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getResults, getNotebookUrl, EffectRow } from '../../../services/api';
import { ForestPlot } from './ForestPlot';

const DIRECTION_TONE: Record<string, string> = {
  positive: 'text-mint',
  negative: 'text-rose',
  null: 'text-ink-secondary',
  mixed: 'text-amber',
};

const Caption = ({ children }: { children: string }) => (
  <p className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-tertiary">{children}</p>
);

const num = (n: number) => (Number.isFinite(n) ? Math.round(n).toLocaleString() : '—');

export function ResultsView({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  const { data, isLoading, isError } = useQuery({
    queryKey: ['results', jobId],
    queryFn: () => getResults(jobId),
    enabled: !!jobId,
  });

  const effects: EffectRow[] = (data?.treatment_effects ?? []).map((te) => ({
    method: te.method,
    estimand: te.estimand,
    estimate: te.estimate,
    ci_lower: te.ci_lower,
    ci_upper: te.ci_upper,
    p_value: te.p_value ?? null,
  }));

  const es = data?.executive_summary;
  const mc = data?.method_consensus;
  const sens = data?.sensitivity_analysis ?? [];
  const recs = data?.recommendations ?? [];

  return (
    <div className="terminal fixed inset-0 z-40 bg-canvas flex flex-col">
      <div className="flex items-center justify-between h-10 px-3 bg-canvas-raised border-b border-edge-subtle shrink-0">
        <span className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-tertiary">[ results ]</span>
        <button
          onClick={onClose}
          className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-secondary hover:text-ink"
        >
          esc · close
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-6 w-full max-w-3xl mx-auto space-y-6">
        {isLoading && <p className="text-xs font-mono text-ink-tertiary">loading results…</p>}
        {isError && <p className="text-xs font-mono text-rose">failed to load results</p>}

        {data && (
          <>
            <div className="text-xs font-mono text-ink-tertiary">
              {data.treatment_variable ?? 'treatment'} → {data.outcome_variable ?? 'outcome'}
            </div>

            {es && (
              <section className="space-y-2">
                <Caption>executive summary</Caption>
                <p className={`text-base font-medium ${DIRECTION_TONE[es.effect_direction] ?? 'text-ink'}`}>
                  {es.headline}
                </p>
                <div className="flex gap-2 text-2xs font-mono">
                  <span className="px-1.5 py-0.5 border border-edge-subtle text-ink-secondary">
                    direction: {es.effect_direction}
                  </span>
                  <span className="px-1.5 py-0.5 border border-edge-subtle text-ink-secondary">
                    confidence: {es.confidence_level}
                  </span>
                </div>
                {es.key_findings.length > 0 && (
                  <ul className="list-disc pl-5 text-xs text-ink-secondary space-y-1">
                    {es.key_findings.map((k, i) => (
                      <li key={i}>{k}</li>
                    ))}
                  </ul>
                )}
              </section>
            )}

            <section className="space-y-2">
              <Caption>treatment effects</Caption>
              {effects.length > 0 ? (
                <ForestPlot effects={effects} />
              ) : (
                <p className="text-xs text-ink-tertiary">no effects produced</p>
              )}
              {mc && (
                <p className="text-2xs font-mono text-ink-tertiary">
                  {mc.n_methods} methods · median {num(mc.median_estimate)} · range [{num(mc.estimate_range[0])},{' '}
                  {num(mc.estimate_range[1])}] · {mc.consensus_strength} consensus
                </p>
              )}
            </section>

            {sens.length > 0 && (
              <section className="space-y-2">
                <Caption>sensitivity</Caption>
                <ul className="space-y-1 text-xs text-ink-secondary">
                  {sens.map((s, i) => (
                    <li key={i}>
                      <span className="font-mono text-bone">{s.method}</span>: {s.interpretation}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {recs.length > 0 && (
              <section className="space-y-2">
                <Caption>recommendations</Caption>
                <ul className="list-disc pl-5 text-xs text-ink-secondary space-y-1">
                  {recs.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </section>
            )}

            <section className="pt-2">
              <a
                href={getNotebookUrl(jobId)}
                className="inline-block px-3 py-2 bg-mint text-ink-inverse text-sm font-medium rounded-md hover:bg-mint-bright transition-colors"
              >
                Download notebook
              </a>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
