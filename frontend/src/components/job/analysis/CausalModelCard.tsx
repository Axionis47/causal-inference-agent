// The causal model the analysis committed to at S3, shown persistently so the
// analyst can see how treatment, outcome, adjusted-for covariates, and any
// suspected latent confounders relate. Renders the shared DagSvg from the
// analysis payload; the adjustment set and latent verdict are the backend's
// graph-derived values (backdoor criterion), not recomputed here.

import type { CausalDagView } from '../../../services/api';
import { Caption } from '../terminal/atoms';
import { DagSvg } from '../terminal/DagSvg';

function Dot({ className }: { className: string }) {
  return <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${className}`} />;
}

export function CausalModelCard({ dag }: { dag: CausalDagView }) {
  const latent = dag.nodes.filter((n) => !n.observed).map((n) => n.name);

  // An empty graph (no nodes) is nothing to draw; skip the card rather than
  // render an empty SVG.
  if (dag.nodes.length === 0) return null;

  return (
    <section className="bg-canvas-raised border border-edge-subtle rounded-md p-3 space-y-2">
      <Caption>[ causal model ]</Caption>

      <div className="h-72">
        <DagSvg
          nodes={dag.nodes.map((n) => n.name)}
          edges={dag.edges}
          treatment={dag.treatment}
          outcome={dag.outcome}
          adjustmentSet={dag.adjustment_set}
          latent={latent}
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-2xs text-ink-tertiary">
        <span className="flex items-center gap-1">
          <Dot className="bg-amber" /> treatment
        </span>
        <span className="flex items-center gap-1">
          <Dot className="bg-mint" /> outcome
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full border border-indigo border-dashed shrink-0" />{' '}
          adjusted
        </span>
        {latent.length > 0 && (
          <span className="flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full border border-rose border-dashed shrink-0" />{' '}
            suspected latent
          </span>
        )}
        <span className="ml-auto">
          estimand <span className="text-ink-secondary">{dag.estimand}</span>
        </span>
      </div>

      {dag.latent_confounding && (
        <p className="text-2xs font-mono text-rose">
          {latent.length} suspected latent confounder{latent.length === 1 ? '' : 's'} on an open
          backdoor path; the effect is not point-identified (see the claim limitations).
        </p>
      )}
    </section>
  );
}
