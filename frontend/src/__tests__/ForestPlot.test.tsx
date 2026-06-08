import { describe, it, expect } from 'vitest';
import { dedupeEffects } from '../components/job/terminal/ForestPlot';
import type { EffectRow } from '../services/api';

const ols = (estimate: number): EffectRow => ({
  method: 'OLS Regression',
  estimand: 'ATE',
  estimate,
  ci_lower: estimate - 100,
  ci_upper: estimate + 100,
  p_value: 0.2,
});

describe('dedupeEffects', () => {
  it('collapses byte-identical rows', () => {
    // The real LaLonde run emitted the same OLS fit 7 times.
    const rows = [ols(302.46), ols(302.46), ols(302.46)];
    expect(dedupeEffects(rows)).toHaveLength(1);
  });

  it('keeps distinct estimates of the same method', () => {
    // Different specifications produce different OLS estimates; those are real
    // and must not be collapsed.
    const rows = [ols(236.1), ols(302.46), ols(939.0)];
    expect(dedupeEffects(rows)).toHaveLength(3);
  });

  it('preserves first-seen order', () => {
    const rows = [ols(939.0), ols(236.1), ols(939.0)];
    expect(dedupeEffects(rows).map((e) => e.estimate)).toEqual([939.0, 236.1]);
  });

  it('distinguishes rows that differ only by method or estimand', () => {
    const ipw: EffectRow = { ...ols(302.46), method: 'Inverse Probability Weighting' };
    const att: EffectRow = { ...ols(302.46), estimand: 'ATT' };
    expect(dedupeEffects([ols(302.46), ipw, att])).toHaveLength(3);
  });
});
