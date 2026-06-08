// Per-agent reasoning steps for the FocusPane: the THINK -> ACT -> OBSERVE diary
// of the focused agent, read from /jobs/:id/traces. Terminal-styled so it sits
// under the agent's finding/challenge. Each step shows the reasoning (THINK),
// the tools called (ACT), and a one-line result preview (OBSERVE).

import type { AgentTrace } from '../../../services/api';

function stepTokens(t: AgentTrace): number {
  return (t.token_usage?.input_tokens || 0) + (t.token_usage?.output_tokens || 0);
}

function stepDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

/** A compact one-line OBSERVE preview: the string output if present, else a
 *  compact JSON of the outputs map. Null when there is nothing to show. */
function observeLine(outputs: Record<string, unknown>): string | null {
  if (!outputs || Object.keys(outputs).length === 0) return null;
  const out = (outputs as { output?: unknown }).output;
  if (typeof out === 'string' && out.trim()) return out;
  try {
    return JSON.stringify(outputs);
  } catch {
    return null;
  }
}

export function TraceSteps({ traces }: { traces: AgentTrace[] }) {
  return (
    <ol className="space-y-3">
      {traces.map((t, i) => {
        const tok = stepTokens(t);
        const obs = observeLine(t.outputs);
        return (
          <li key={`${t.action}-${t.timestamp}-${i}`} className="space-y-1">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-2xs font-mono text-indigo uppercase tracking-[0.1em] truncate">
                {t.action}
              </span>
              <span className="text-2xs font-mono text-ink-tertiary tabular shrink-0">
                {stepDuration(t.duration_ms)}
                {tok > 0 ? ` · ${tok}tok` : ''}
              </span>
            </div>
            {t.reasoning && (
              <p className="text-xs font-mono text-ink-secondary whitespace-pre-wrap">
                {t.reasoning}
              </p>
            )}
            {t.tools_called?.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {t.tools_called.map((tool, j) => (
                  <span
                    key={j}
                    className="text-2xs font-mono text-indigo border border-indigo/40 px-1.5 py-0.5"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            )}
            {obs && <p className="text-2xs font-mono text-ink-tertiary truncate">{obs}</p>}
          </li>
        );
      })}
    </ol>
  );
}
