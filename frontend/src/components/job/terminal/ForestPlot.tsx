// A small dependency-free forest plot: one row per estimation method, the point
// at the estimate, a line across its 95% CI, and a dashed line at the null
// effect (0). Methods whose CI excludes 0 (p < 0.05) are amber.

import { EffectRow } from '../../../services/api';

const W = 380;
const ML = 88; // left margin for method labels
const MR = 44; // right margin for the value
const ROW = 26;
const TOP = 8;

/** Drop exact-duplicate rows. The estimator can emit the same fit several times
 *  (e.g. 7 identical OLS rows on a real run), which stacks indistinguishable
 *  points on the plot. Identity = method + estimand + estimate + CI + p, so
 *  distinct estimates are always kept; only byte-identical repeats collapse. */
export function dedupeEffects(effects: EffectRow[]): EffectRow[] {
  const seen = new Set<string>();
  const out: EffectRow[] = [];
  for (const e of effects) {
    const key = [e.method, e.estimand, e.estimate, e.ci_lower, e.ci_upper, e.p_value].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(e);
  }
  return out;
}

export function ForestPlot({ effects }: { effects: EffectRow[] }) {
  const rows = dedupeEffects(effects);
  if (rows.length === 0) return null;

  let lo = Math.min(0, ...rows.map((e) => e.ci_lower));
  let hi = Math.max(0, ...rows.map((e) => e.ci_upper));
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.08 || 1;
  lo -= pad;
  hi += pad;
  const x = (v: number) => ML + ((v - lo) / (hi - lo)) * (W - ML - MR);
  const H = TOP * 2 + rows.length * ROW;
  const zero = x(0);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="forest plot of effect estimates">
      <line
        x1={zero}
        y1={TOP}
        x2={zero}
        y2={TOP + rows.length * ROW}
        className="stroke-edge-strong"
        strokeDasharray="3 3"
        strokeWidth={1}
      />
      {rows.map((e, i) => {
        const cy = TOP + i * ROW + ROW / 2;
        const sig = e.p_value != null && e.p_value < 0.05;
        return (
          <g key={`${e.method}-${i}`}>
            <text x={4} y={cy + 3} className="fill-ink-secondary font-mono" style={{ fontSize: '9px' }}>
              {e.method}
            </text>
            <line x1={x(e.ci_lower)} y1={cy} x2={x(e.ci_upper)} y2={cy} className="stroke-edge-strong" strokeWidth={1} />
            <line x1={x(e.ci_lower)} y1={cy - 3} x2={x(e.ci_lower)} y2={cy + 3} className="stroke-edge-strong" strokeWidth={1} />
            <line x1={x(e.ci_upper)} y1={cy - 3} x2={x(e.ci_upper)} y2={cy + 3} className="stroke-edge-strong" strokeWidth={1} />
            <circle cx={x(e.estimate)} cy={cy} r={3.5} className={sig ? 'fill-amber' : 'fill-ink-tertiary'} />
            <text x={W - MR + 4} y={cy + 3} className="fill-ink font-mono" style={{ fontSize: '9px' }}>
              {e.estimate.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
