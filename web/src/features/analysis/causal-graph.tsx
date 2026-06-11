import { useId } from "react";
import { EvidenceLinks } from "./inspector";
import type { components } from "@/lib/api/schema";

type Frame = components["schemas"]["Frame"];

export function CausalGraph({
  frame,
  inspect,
}: {
  frame: Frame;
  inspect: (key: string) => void;
}) {
  const nodes = (frame.causal.nodes ?? []).map((node) => ({
    ...node,
    roles: node.roles ?? [],
  }));
  const edges = frame.causal.edges ?? [];
  const arrow = useId().replaceAll(":", "");
  if (!nodes.length)
    return (
      <p className="subtext">
        No causal graph can be built from the supplied information yet.
      </p>
    );
  const depth = new Map(nodes.map((node) => [node.id, 0]));
  for (let i = 0; i < nodes.length; i++) {
    for (const edge of edges)
      depth.set(
        edge.target,
        Math.max(
          depth.get(edge.target) ?? 0,
          (depth.get(edge.source) ?? 0) + 1,
        ),
      );
  }
  const levels = [...new Set(depth.values())].sort((a, b) => a - b);
  const width = Math.max(
    600,
    ...levels.map(
      (level) => nodes.filter((n) => depth.get(n.id) === level).length * 200,
    ),
  );
  const height = levels.length * 105 + 30;
  const positions = new Map(
    nodes.map((node) => {
      const level = depth.get(node.id) ?? 0;
      const peers = nodes.filter((n) => depth.get(n.id) === level);
      return [
        node.id,
        {
          x:
            ((peers.findIndex((n) => n.id === node.id) + 0.5) * width) /
            peers.length,
          y: level * 105 + 50,
        },
      ];
    }),
  );
  return (
    <div className="causal-graph">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Proposed causal directed acyclic graph"
      >
        <defs>
          <marker
            id={arrow}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>
        {edges.map((edge) => {
          const from = positions.get(edge.source)!;
          const to = positions.get(edge.target)!;
          return (
            <path
              key={`${edge.source}:${edge.target}`}
              d={`M ${from.x} ${from.y + 24} L ${to.x} ${to.y - 28}`}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeDasharray={edge.basis === "assumption" ? "5 4" : undefined}
              markerEnd={`url(#${arrow})`}
            >
              <title>{edge.explanation}</title>
            </path>
          );
        })}
        {nodes.map((node) => {
          const point = positions.get(node.id)!;
          return (
            <g key={node.id} transform={`translate(${point.x},${point.y})`}>
              <title>
                {node.label} ({node.roles.join(", ")})
              </title>
              <rect
                x="-88"
                y="-25"
                width="176"
                height="50"
                rx="3"
                className={`graph-node ${node.roles.map((role) => `graph-node-${role}`).join(" ")}`}
              />
              <text
                textAnchor="middle"
                y="-3"
                fontSize="12"
                fill="currentColor"
              >
                {node.label.length > 21
                  ? node.label.slice(0, 20) + "…"
                  : node.label}
              </text>
              <text textAnchor="middle" y="15" fontSize="9" fill="currentColor">
                {node.roles.join(" / ").toUpperCase()}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="subtext">
        Dashed arrows are working assumptions. Direction comes from the semantic
        account, never from correlation alone.
      </p>
      <details className="mt-3">
        <summary>Edges and evidence</summary>
        <ul className="analysis-list">
          {edges.map((edge) => (
            <li key={`${edge.source}:${edge.target}`}>
              <strong>
                {nodes.find((n) => n.id === edge.source)?.label} →{" "}
                {nodes.find((n) => n.id === edge.target)?.label}
              </strong>
              <p>{edge.explanation}</p>
              <p className="subtext">{edge.basis}</p>
              <EvidenceLinks references={edge.references} inspect={inspect} />
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
