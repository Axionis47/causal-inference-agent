import { memo, useMemo } from 'react';
import { CausalGraph } from '../../services/api';

interface CausalGraphViewProps {
  graph: CausalGraph;
  treatmentVariable?: string;
  outcomeVariable?: string;
}

interface NodePos {
  id: string;
  x: number;
  y: number;
  fill: string;
}

const NODE_R = 24;
const SVG_W = 600;
const SVG_H = 340;
const LABEL_OFFSET = 18;

function layoutNodes(
  nodes: string[],
  _edges: CausalGraph['edges'],
  treatment?: string,
  outcome?: string,
): NodePos[] {
  if (nodes.length === 0) return [];

  const treat = treatment ?? nodes[0];
  const out = outcome ?? nodes[nodes.length - 1];
  const others = nodes.filter((n) => n !== treat && n !== out);

  const positions: NodePos[] = [];
  const pad = 80;
  const innerW = SVG_W - pad * 2;

  // Treatment on left, outcome on right
  positions.push({ id: treat, x: pad, y: SVG_H / 2, fill: '#22c55e' });
  positions.push({ id: out, x: SVG_W - pad, y: SVG_H / 2, fill: '#ef4444' });

  // Confounders / other nodes spread across top region
  if (others.length === 1) {
    positions.push({ id: others[0], x: SVG_W / 2, y: pad, fill: '#6b7280' });
  } else {
    others.forEach((n, i) => {
      const t = others.length > 1 ? i / (others.length - 1) : 0.5;
      const x = pad + innerW * 0.15 + t * innerW * 0.7;
      const row = i % 2 === 0 ? pad : pad + 50;
      positions.push({ id: n, x, y: row, fill: '#6b7280' });
    });
  }

  return positions;
}

function CausalGraphView({ graph, treatmentVariable, outcomeVariable }: CausalGraphViewProps) {
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];

  const positions = useMemo(
    () => layoutNodes(nodes, edges, treatmentVariable, outcomeVariable),
    [nodes, edges, treatmentVariable, outcomeVariable],
  );

  const posMap = useMemo(() => {
    const m = new Map<string, NodePos>();
    positions.forEach((p) => m.set(p.id, p));
    return m;
  }, [positions]);

  if (nodes.length === 0) return null;

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="w-full max-w-2xl mx-auto"
      role="img"
      aria-label="Causal graph visualization"
    >
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="10"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
        </marker>
      </defs>

      {/* Edges */}
      {edges.map((edge, idx) => {
        const src = posMap.get(edge.source);
        const tgt = posMap.get(edge.target);
        if (!src || !tgt) return null;

        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const ux = dx / dist;
        const uy = dy / dist;

        const x1 = src.x + ux * NODE_R;
        const y1 = src.y + uy * NODE_R;
        const x2 = tgt.x - ux * (NODE_R + 4);
        const y2 = tgt.y - uy * (NODE_R + 4);

        return (
          <line
            key={idx}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="#9ca3af"
            strokeWidth={1.5}
            markerEnd="url(#arrowhead)"
          />
        );
      })}

      {/* Nodes */}
      {positions.map((node) => (
        <g key={node.id}>
          <circle
            cx={node.x}
            cy={node.y}
            r={NODE_R}
            fill={node.fill}
            opacity={0.85}
          />
          <text
            x={node.x}
            y={node.y + NODE_R + LABEL_OFFSET}
            textAnchor="middle"
            fontSize="11"
            fill="#374151"
            fontWeight={500}
          >
            {node.id.length > 18 ? node.id.slice(0, 16) + '...' : node.id}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default memo(CausalGraphView);
