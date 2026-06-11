// One mono line of run totals: tokens in/out, cost (4 decimals), tool calls.

import type { AnalysisCosts } from '../../../services/api';

export function CostLine({ costs }: { costs: AnalysisCosts }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-2xs text-ink-tertiary tabular">
      <span>
        tok in{' '}
        <span className="text-ink-secondary">
          {costs.total_input_tokens.toLocaleString()}
        </span>
      </span>
      <span>
        tok out{' '}
        <span className="text-ink-secondary">
          {costs.total_output_tokens.toLocaleString()}
        </span>
      </span>
      <span>
        cost{' '}
        <span className="text-ink-secondary">${costs.total_cost_usd.toFixed(4)}</span>
      </span>
      <span>
        tool calls{' '}
        <span className="text-ink-secondary">
          {costs.total_tool_calls.toLocaleString()}
        </span>
      </span>
    </div>
  );
}
