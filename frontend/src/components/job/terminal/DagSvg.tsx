// A small, dependency-free SVG renderer for a causal DAG. Nodes are laid out on
// a circle (readable for the handful of variables a causal DAG carries); the
// treatment is amber, the outcome mint, and adjusted-for variables get an indigo
// ring. Suspected latent (unobserved) confounders are drawn hollow with a dashed
// rose outline. Directed edges are solid arrows, undirected/bidirected dashed.
//
// Dense graphs (a wide adjustment set can run to dozens of nodes) stay legible
// through zoom and pan: the wheel scales the viewBox anchored at the cursor, drag
// pans it, and the +/-/reset controls do the same without a mouse. The viewBox is
// the whole mechanism; no dependency.

import { useEffect, useRef, useState } from 'react';
import { DagEdgeView } from '../../../services/api';

const W = 340;
const H = 260;
const NODE_R = 11;
const MIN_W = W / 8; // deepest zoom in
const MAX_W = W * 1.5; // furthest zoom out
const WHEEL_STEP = 1.12;
const BUTTON_STEP = 0.7; // < 1 zooms in

type Box = { x: number; y: number; w: number; h: number };

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

export function DagSvg({
  nodes,
  edges,
  treatment,
  outcome,
  adjustmentSet,
  latent = [],
}: {
  nodes: string[];
  edges: DagEdgeView[];
  treatment: string | null;
  outcome: string | null;
  adjustmentSet: string[];
  latent?: string[];
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
  const latentSet = new Set(latent);

  // viewBox-driven zoom/pan state. The full graph is {0, 0, W, H}.
  const [box, setBox] = useState<Box>({ x: 0, y: 0, w: W, h: H });
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ cx: number; cy: number } | null>(null);

  const reset = () => setBox({ x: 0, y: 0, w: W, h: H });

  // Re-frame when the graph identity changes (node count is a cheap proxy) so a
  // newly loaded run starts framed rather than at the previous run's zoom.
  useEffect(() => {
    setBox({ x: 0, y: 0, w: W, h: H });
  }, [nodes.length]);

  // Zoom around a point given as 0..1 fractions of the viewport, keeping aspect.
  const zoomAt = (factor: number, fx: number, fy: number) =>
    setBox((prev) => {
      const w = clamp(prev.w * factor, MIN_W, MAX_W);
      const h = w * (H / W);
      const px = prev.x + fx * prev.w;
      const py = prev.y + fy * prev.h;
      return { w, h, x: px - fx * w, y: py - fy * h };
    });

  const zoomCenter = (factor: number) => zoomAt(factor, 0.5, 0.5);

  // Wheel zoom needs a non-passive listener to preventDefault (React's onWheel is
  // passive), so attach it natively. zoomAt uses a functional update, so the
  // once-attached handler never goes stale.
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const fx = (e.clientX - rect.left) / rect.width;
      const fy = (e.clientY - rect.top) / rect.height;
      zoomAt(e.deltaY < 0 ? 1 / WHEEL_STEP : WHEEL_STEP, fx, fy);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    drag.current = { cx: e.clientX, cy: e.clientY };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!drag.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dx = (e.clientX - drag.current.cx) / rect.width;
    const dy = (e.clientY - drag.current.cy) / rect.height;
    drag.current = { cx: e.clientX, cy: e.clientY };
    setBox((prev) => ({ ...prev, x: prev.x - dx * prev.w, y: prev.y - dy * prev.h }));
  };
  const endDrag = () => {
    drag.current = null;
  };

  const ctl =
    'w-5 h-5 flex items-center justify-center border border-edge bg-canvas-raised ' +
    'text-ink-secondary hover:text-ink hover:border-edge-strong rounded-sm leading-none';

  return (
    <div className="relative w-full h-full">
      <svg
        ref={svgRef}
        viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`}
        className="w-full h-full cursor-grab active:cursor-grabbing touch-none select-none"
        role="img"
        aria-label="causal dag"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
      >
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
          const isLatent = latentSet.has(id);
          const fill = isLatent
            ? 'fill-none'
            : id === treatment
              ? 'fill-amber'
              : id === outcome
                ? 'fill-mint'
                : 'fill-canvas-inset';
          const stroke = isLatent ? 'stroke-rose' : 'stroke-edge-strong';
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
              <circle
                cx={p.x}
                cy={p.y}
                r={NODE_R}
                className={`${fill} ${stroke}`}
                strokeWidth={1}
                strokeDasharray={isLatent ? '3 2' : undefined}
              />
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

      <div className="absolute top-1 right-1 flex gap-1 font-mono text-2xs">
        <button type="button" aria-label="zoom in" className={ctl} onClick={() => zoomCenter(BUTTON_STEP)}>
          +
        </button>
        <button
          type="button"
          aria-label="zoom out"
          className={ctl}
          onClick={() => zoomCenter(1 / BUTTON_STEP)}
        >
          &minus;
        </button>
        <button type="button" aria-label="reset view" className={`${ctl} w-auto px-1`} onClick={reset}>
          reset
        </button>
      </div>
    </div>
  );
}
