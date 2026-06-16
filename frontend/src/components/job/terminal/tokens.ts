// Pure derivation of the TopBar token total. No React, no I/O.
import type { AgentTrace, AnalysisAgentView } from '../../../services/api';

/**
 * Total tokens for the run header, rederived additively each render: sum every
 * agent's input+output so the header equals the sum of the per-agent panels and
 * can never drift from a separately stored aggregate. v2 runs use the analysis
 * agents; legacy jobs with no analysis run fall back to the trace token_usage
 * sum. Returns null when nothing is known yet (so the header shows "—").
 */
export function deriveTokenTotal(
  agents: AnalysisAgentView[] | undefined,
  traces: AgentTrace[] | undefined,
): number | null {
  if (agents && agents.length > 0) {
    const total = agents.reduce(
      (sum, a) => sum + (a.tokens?.input_tokens || 0) + (a.tokens?.output_tokens || 0),
      0,
    );
    return total > 0 ? total : null;
  }
  if (!traces || traces.length === 0) return null;
  return traces.reduce(
    (sum, t) => sum + (t.token_usage?.input_tokens || 0) + (t.token_usage?.output_tokens || 0),
    0,
  );
}
