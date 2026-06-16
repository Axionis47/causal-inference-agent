import { describe, it, expect } from 'vitest';
import { deriveTokenTotal } from '../components/job/terminal/tokens';
import type { AgentTrace, AnalysisAgentView } from '../services/api';

const agent = (input: number, output: number): AnalysisAgentView =>
  ({ tokens: { input_tokens: input, output_tokens: output } } as AnalysisAgentView);

const trace = (input: number, output: number): AgentTrace =>
  ({ token_usage: { input_tokens: input, output_tokens: output } } as AgentTrace);

describe('deriveTokenTotal', () => {
  it('sums input+output across the v2 analysis agents', () => {
    const agents = [agent(3008, 472), agent(10087, 3733), agent(0, 0)];
    expect(deriveTokenTotal(agents, undefined)).toBe(3008 + 472 + 10087 + 3733);
  });

  it('prefers the analysis agents over legacy traces', () => {
    expect(deriveTokenTotal([agent(100, 50)], [trace(9999, 9999)])).toBe(150);
  });

  it('falls back to the trace token_usage sum when there is no analysis run', () => {
    expect(deriveTokenTotal(undefined, [trace(200, 30), trace(10, 5)])).toBe(245);
  });

  it('returns null when nothing is known yet', () => {
    expect(deriveTokenTotal(undefined, undefined)).toBeNull();
    expect(deriveTokenTotal([], [])).toBeNull();
  });

  it('returns null when the agents exist but recorded zero tokens', () => {
    expect(deriveTokenTotal([agent(0, 0)], undefined)).toBeNull();
  });
});
