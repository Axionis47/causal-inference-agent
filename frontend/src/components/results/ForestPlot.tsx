import { memo, useMemo } from 'react';
import { TreatmentEffect } from '../../services/api';
import { SIGNIFICANCE_ALPHA } from '../../config/constants';

interface ForestPlotProps {
  effects: TreatmentEffect[];
}

function ForestPlot({ effects }: ForestPlotProps) {
  const valid = useMemo(
    () =>
      effects
        .filter((e) => isFinite(e.estimate) && isFinite(e.ci_lower) && isFinite(e.ci_upper))
        .sort((a, b) => a.estimate - b.estimate),
    [effects]
  );

  if (valid.length === 0) return null;

  const ROW = 36, PAD_TOP = 24, PAD_BOT = 32, LEFT = 130, RIGHT = 70, W = 600;
  const plotW = W - LEFT - RIGHT;
  const H = PAD_TOP + valid.length * ROW + PAD_BOT;
  const axisY = PAD_TOP + valid.length * ROW;

  const { xMin, xMax, toX } = useMemo(() => {
    const lo = Math.min(...valid.map((e) => e.ci_lower));
    const hi = Math.max(...valid.map((e) => e.ci_upper));
    const pad = (hi - lo || 1) * 0.15;
    const mn = Math.min(lo - pad, -pad * 0.5);
    const mx = Math.max(hi + pad, pad * 0.5);
    return { xMin: mn, xMax: mx, toX: (v: number) => LEFT + ((v - mn) / (mx - mn)) * plotW };
  }, [valid, plotW]);

  const zeroX = toX(0);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
      aria-label="Forest plot of treatment effect estimates">
      {/* Zero reference line */}
      <line x1={zeroX} y1={PAD_TOP - 4} x2={zeroX} y2={axisY + 4}
        stroke="#d1d5db" strokeWidth={1} strokeDasharray="4 3" />

      {valid.map((effect, i) => {
        const y = PAD_TOP + i * ROW + ROW / 2;
        const sig = effect.p_value != null && effect.p_value < SIGNIFICANCE_ALPHA;
        const color = sig ? '#1f2937' : '#9ca3af';
        const x1 = toX(effect.ci_lower), x2 = toX(effect.ci_upper), cx = toX(effect.estimate);
        return (
          <g key={`${effect.method}-${i}`}>
            {i % 2 === 1 && <rect x={0} y={y - ROW / 2} width={W} height={ROW} fill="#f9fafb" />}
            <text x={LEFT - 10} y={y} textAnchor="end" dominantBaseline="central"
              fontSize={12} fill="#374151" fontFamily="system-ui, sans-serif">
              {effect.method}
            </text>
            <line x1={x1} y1={y} x2={x2} y2={y} stroke={color} strokeWidth={2} />
            <line x1={x1} y1={y - 5} x2={x1} y2={y + 5} stroke={color} strokeWidth={1.5} />
            <line x1={x2} y1={y - 5} x2={x2} y2={y + 5} stroke={color} strokeWidth={1.5} />
            <circle cx={cx} cy={y} r={4.5} fill={color} />
            <text x={LEFT + plotW + 10} y={y} dominantBaseline="central"
              fontSize={11} fill={color} fontFamily="ui-monospace, monospace"
              fontWeight={sig ? 600 : 400}>
              {effect.estimate.toFixed(3)}
            </text>
          </g>
        );
      })}

      {/* X-axis */}
      <line x1={LEFT} y1={axisY + 8} x2={LEFT + plotW} y2={axisY + 8}
        stroke="#9ca3af" strokeWidth={0.75} />
      <text x={LEFT + plotW / 2} y={H - 4} textAnchor="middle"
        fontSize={11} fill="#6b7280" fontFamily="system-ui, sans-serif">
        Treatment Effect Estimate
      </text>
      <text x={LEFT} y={axisY + 20} textAnchor="middle"
        fontSize={9} fill="#9ca3af" fontFamily="ui-monospace, monospace">
        {xMin.toFixed(2)}
      </text>
      <text x={LEFT + plotW} y={axisY + 20} textAnchor="middle"
        fontSize={9} fill="#9ca3af" fontFamily="ui-monospace, monospace">
        {xMax.toFixed(2)}
      </text>
      {zeroX > LEFT + 20 && zeroX < LEFT + plotW - 20 && (
        <text x={zeroX} y={axisY + 20} textAnchor="middle"
          fontSize={9} fill="#9ca3af" fontFamily="ui-monospace, monospace">
          0
        </text>
      )}
    </svg>
  );
}

export default memo(ForestPlot);
