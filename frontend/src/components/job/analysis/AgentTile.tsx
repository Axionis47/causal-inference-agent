// One tile per analysis agent. Pure presentation: typed payload in, no I/O.
// Status colours follow the live/settled axis (aesthetics.md §2/§6):
// waiting = edge-subtle (dim), running = amber pulse, passed = mint,
// warning = amber static, failed = rose.

import { useState } from 'react';
import type {
  AnalysisAgentStatus,
  AnalysisAgentView,
} from '../../../services/api';
import { formatSeconds } from '../terminal/format';

const DOT_CLS: Record<AnalysisAgentStatus, string> = {
  waiting: 'bg-edge-subtle',
  running: 'bg-amber animate-pulse-live',
  passed: 'bg-mint',
  warning: 'bg-amber',
  failed: 'bg-rose',
};

const CHIP_CLS: Record<AnalysisAgentStatus, string> = {
  waiting: 'text-ink-tertiary',
  running: 'text-amber',
  passed: 'text-mint',
  warning: 'text-amber',
  failed: 'text-rose',
};

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';

export function AgentTile({ agent }: { agent: AnalysisAgentView }) {
  const [expanded, setExpanded] = useState(false);
  const isWaiting = agent.status === 'waiting';

  return (
    <div
      className={`bg-canvas-raised border border-edge-subtle rounded-md p-3 flex flex-col gap-2 min-w-0 ${
        agent.status === 'running' ? 'border-amber/40' : ''
      }`}
      data-testid={`agent-tile-${agent.agent}`}
    >
      {/* header: dot + name + status chip */}
      <div className="flex items-center gap-2 min-w-0">
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${DOT_CLS[agent.status]}`}
        />
        <span
          className={`font-mono text-xs truncate ${
            isWaiting ? 'text-ink-tertiary' : 'text-ink'
          }`}
        >
          {agent.agent}
        </span>
        <span className={`${labelCls} ml-auto shrink-0 ${CHIP_CLS[agent.status]}`}>
          {agent.status}
        </span>
      </div>

      {/* the live step while the agent runs */}
      {agent.current_step && (
        <p className="font-mono text-2xs text-amber truncate">{agent.current_step}</p>
      )}

      {/* public summary, clamped to ~4 lines with an expand toggle */}
      {agent.public_summary && (
        <div className="min-w-0">
          <p
            className={`text-xs text-ink-secondary whitespace-pre-wrap ${
              expanded ? '' : 'line-clamp-4'
            }`}
          >
            {agent.public_summary}
          </p>
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="text-2xs font-mono text-ink-tertiary hover:text-ink underline underline-offset-2 mt-0.5"
          >
            {expanded ? 'less' : 'more'}
          </button>
        </div>
      )}

      {/* warnings */}
      {agent.warnings.length > 0 && (
        <ul className="space-y-0.5">
          {agent.warnings.map((w, i) => (
            <li key={i} className="text-2xs font-mono text-amber">
              ! {w}
            </li>
          ))}
        </ul>
      )}

      {/* footer: tokens / cost / elapsed / tools / artifacts */}
      <div className="mt-auto pt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 font-mono text-2xs text-ink-tertiary tabular">
        <span>
          tok {agent.tokens.input_tokens.toLocaleString()}/
          {agent.tokens.output_tokens.toLocaleString()}
        </span>
        <span>${agent.cost_usd.toFixed(4)}</span>
        <span>{formatSeconds(agent.elapsed_seconds)}</span>
        <span>{agent.tool_call_count} tools</span>
        <span>{agent.artifact_ids.length} artifacts</span>
        {agent.attempt > 1 && <span className="text-amber">attempt {agent.attempt}</span>}
      </div>
    </div>
  );
}
