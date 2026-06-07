// A small dependency-free Love plot: one row per covariate, a bar to its
// absolute standardized mean difference (SMD), and a dashed line at the 0.1
// imbalance threshold. Covariates over the threshold are rose, balanced ones mint.

import { BalanceRow } from '../../../services/api';

const W = 380;
const ML = 88;
const MR = 10;
const ROW = 20;
const TOP = 8;

export function BalancePlot({ balance }: { balance: BalanceRow[] }) {
  const rows = balance.filter((b): b is { covariate: string; smd: number } => b.smd != null);
  if (rows.length === 0) return null;

  const maxSmd = Math.max(0.25, ...rows.map((r) => Math.abs(r.smd)));
  const x = (v: number) => ML + (Math.abs(v) / maxSmd) * (W - ML - MR);
  const H = TOP * 2 + rows.length * ROW;
  const thr = x(0.1);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="covariate balance">
      <line
        x1={thr}
        y1={TOP}
        x2={thr}
        y2={TOP + rows.length * ROW}
        className="stroke-rose/50"
        strokeDasharray="3 3"
        strokeWidth={1}
      />
      {rows.map((r, i) => {
        const cy = TOP + i * ROW + ROW / 2;
        const over = Math.abs(r.smd) > 0.1;
        return (
          <g key={`${r.covariate}-${i}`}>
            <text x={4} y={cy + 3} className="fill-ink-secondary font-mono" style={{ fontSize: '9px' }}>
              {r.covariate}
            </text>
            <line x1={ML} y1={cy} x2={x(r.smd)} y2={cy} className={over ? 'stroke-rose' : 'stroke-mint'} strokeWidth={3} />
            <circle cx={x(r.smd)} cy={cy} r={2.5} className={over ? 'fill-rose' : 'fill-mint'} />
          </g>
        );
      })}
    </svg>
  );
}
