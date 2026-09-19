import { bars, categorical, categories, fmtTick, markX, path, plotBox, points, ticks, yExtent, type FigureSpec } from "../figure";

const COLORS = ["var(--accent)", "var(--declared)", "var(--ink-3)", "var(--accent-ink)"];

// Draws a figure spec (causal_agent/viz/spec.py) as inline SVG. Every drawn value carries its address in a title, so
// hovering a bar or a point shows what the chat can cite.
export default function Figure({ spec }: { spec: FigureSpec }) {
  const box = plotBox();
  const yd = yExtent(spec);
  const sy = (v: number) => box.top + box.height - ((v - yd[0]) / (yd[1] - yd[0] || 1)) * box.height;
  const yTicks = ticks(yd);
  const cat = categorical(spec);
  const cats = cat ? categories(spec) : [];
  const addr = (si: number, i: number) => `figure:${spec.id}.${spec.series[si].name.toLowerCase().replace(/[^0-9a-z]+/g, "_").replace(/^_|_$/g, "")}.${i}`;
  const width = box.left + box.width + 12;
  const height = box.top + box.height + 46;
  return (
    <figure className="fig" aria-label={spec.title}>
      <figcaption>
        <span className="addr">figure:{spec.id}</span> {spec.title}
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={box.left} x2={box.left + box.width} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={box.left - 6} y={sy(t)} className="tick" textAnchor="end" dominantBaseline="middle">
              {fmtTick(t)}
            </text>
          </g>
        ))}
        {spec.kind === "bars" &&
          bars(spec, box, yd).map((b) => (
            <rect key={`${b.series}-${b.i}`} x={b.x} y={b.y} width={b.width} height={b.height} fill={COLORS[b.series % COLORS.length]} opacity={0.85}>
              <title>{`[${addr(b.series, b.i)}] ${spec.series[b.series].name} · ${b.label}: ${fmtTick(b.value)}${spec.series[b.series].n ? ` (n=${spec.series[b.series].n![b.i]})` : ""}`}</title>
            </rect>
          ))}
        {spec.kind !== "bars" &&
          spec.series.map((s, si) => {
            const ps = points({ ...spec, series: [s] }, box, yd).map((p) => ({ ...p, series: si }));
            return (
              <g key={s.name} fill={COLORS[si % COLORS.length]} stroke={COLORS[si % COLORS.length]}>
                {(spec.kind === "lines" || spec.kind === "density") && <path d={path(ps)} fill="none" strokeWidth={1.6} />}
                {spec.kind === "density" && ps.length > 1 && (
                  <path d={`${path(ps)} L${ps[ps.length - 1].x.toFixed(1)} ${(box.top + box.height).toFixed(1)} L${ps[0].x.toFixed(1)} ${(box.top + box.height).toFixed(1)} Z`} opacity={0.15} stroke="none" />
                )}
                {ps.map((p) => (
                  <g key={p.i}>
                    {p.lo !== undefined && p.hi !== undefined && <line x1={p.x} x2={p.x} y1={p.lo} y2={p.hi} strokeWidth={1.2} />}
                    <circle cx={p.x} cy={p.y} r={spec.kind === "density" ? 1.5 : 3}>
                      <title>{`[${addr(si, p.i)}] ${s.name} · ${p.label}: ${fmtTick(p.value)}${s.n ? ` (n=${s.n[p.i]})` : ""}`}</title>
                    </circle>
                  </g>
                ))}
              </g>
            );
          })}
        {spec.marks.map((m, i) => {
          const x = markX(spec, box, m);
          if (x === null) return null;
          return (
            <g key={`m${i}`} className="mark">
              <line x1={x} x2={x} y1={box.top} y2={box.top + box.height} />
              <text x={x + 4} y={box.top + 10} className="tick">
                {m.label}
              </text>
            </g>
          );
        })}
        <line x1={box.left} x2={box.left + box.width} y1={box.top + box.height} y2={box.top + box.height} className="axis" />
        {cat
          ? cats.map((c, i) => (
              <text key={c} x={box.left + (i + 0.5) * (box.width / cats.length)} y={box.top + box.height + 14} className="tick" textAnchor="middle">
                {c.length > 18 ? `${c.slice(0, 17)}…` : c}
              </text>
            ))
          : ticks([Math.min(...spec.series.flatMap((s) => s.x.map(Number))), Math.max(...spec.series.flatMap((s) => s.x.map(Number)))], 6).map((t) => {
              const xs = points({ ...spec, series: [{ name: "_", x: [t], y: [yd[0]] }] }, box, yd);
              return xs.length ? (
                <text key={t} x={xs[0].x} y={box.top + box.height + 14} className="tick" textAnchor="middle">
                  {fmtTick(t)}
                </text>
              ) : null;
            })}
        {spec.x_label && (
          <text x={box.left + box.width / 2} y={height - 4} className="tick" textAnchor="middle">
            {spec.x_label}
          </text>
        )}
        {spec.y_label && (
          <text x={12} y={box.top + box.height / 2} className="tick" textAnchor="middle" transform={`rotate(-90 12 ${box.top + box.height / 2})`}>
            {spec.y_label}
          </text>
        )}
      </svg>
      <div className="legend">
        {spec.series.map((s, i) => (
          <span key={s.name}>
            <i style={{ background: COLORS[i % COLORS.length] }} /> {s.name}
          </span>
        ))}
      </div>
      {spec.note && <p className="note">{spec.note}</p>}
    </figure>
  );
}
