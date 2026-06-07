// A small, dependency-free SVG renderer for a causal DAG. Nodes are laid out on
// a circle (readable for the handful of variables a causal DAG carries); the
// treatment is amber, the outcome mint, and adjusted-for variables get an indigo
// ring. Directed edges are solid arrows, undirected/bidirected are dashed.

import { DagEdgeView } from '../../../services/api';

const W = 340;
const H = 260;
const NODE_R = 11;

export function DagSvg({
  nodes,
  edges,
  treatment,
  outcome,
  adjustmentSet,
}: {
  nodes: string[];
  edges: DagEdgeView[];
  treatment: string | null;
  outcome: string | null;
  adjustmentSet: string[];
}) {
  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(W, H) / 2 - 44;
  const n = Math.max(nodes.length, 1);

  const pos = new Map<string, { x: number; y: number }>();
  nodes.forEach((id, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
    pos.set(id, { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) });
  });

  const adj = new Set(adjustmentSet);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" role="img" aria-label="causal dag">
      <defs>
        <marker id="dag-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
          <path d="M0,0 L7,3 L0,6 Z" className="fill-ink-tertiary" />
        </marker>
      </defs>

      {edges.map((e, i) => {
        const a = pos.get(e.source);
        const b = pos.get(e.target);
        if (!a || !b) return null;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len;
        const uy = dy / len;
        // Stop the line at the node rim so the arrowhead sits cleanly.
        const x1 = a.x + ux * (NODE_R + 4);
        const y1 = a.y + uy * (NODE_R + 4);
        const x2 = b.x - ux * (NODE_R + 6);
        const y2 = b.y - uy * (NODE_R + 6);
        const directed = e.edge_type === 'directed';
        return (
          <line
            key={`${e.source}-${e.target}-${i}`}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            className="stroke-edge-strong"
            strokeWidth={1}
            markerEnd={directed ? 'url(#dag-arrow)' : undefined}
            strokeDasharray={directed ? undefined : '4 3'}
          />
        );
      })}

      {nodes.map((id) => {
        const p = pos.get(id);
        if (!p) return null;
        const fill =
          id === treatment ? 'fill-amber' : id === outcome ? 'fill-mint' : 'fill-canvas-inset';
        const labelAbove = p.y <= cy;
        return (
          <g key={id}>
            {adj.has(id) && (
              <circle
                cx={p.x}
                cy={p.y}
                r={NODE_R + 4}
                className="fill-none stroke-indigo"
                strokeWidth={1}
                strokeDasharray="3 2"
              />
            )}
            <circle cx={p.x} cy={p.y} r={NODE_R} className={`${fill} stroke-edge-strong`} strokeWidth={1} />
            <text
              x={p.x}
              y={labelAbove ? p.y - NODE_R - 5 : p.y + NODE_R + 11}
              textAnchor="middle"
              className="fill-ink font-mono"
              style={{ fontSize: '9px' }}
            >
              {id}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
