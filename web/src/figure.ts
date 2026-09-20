// A figure is data with addresses (see causal_agent/viz/spec.py). This module turns a spec into the numbers an SVG
// needs: scales, ticks, bar and point positions. Pure functions, tested; Figure.tsx only draws what comes out.

export type Kind = "bars" | "lines" | "points" | "density" | "interval" | "graph";
export type Role = "treatment" | "outcome" | "confounder" | "driver" | "mediator" | "instrument" | "hidden" | "excluded" | "other";

export interface GraphNode {
  id: string;
  label: string;
  role: Role;
}

export interface GraphEdge {
  src: string;
  dst: string;
  cites: string[];
}

export interface Series {
  name: string;
  x: (string | number)[];
  y: (number | null)[];
  lo?: (number | null)[] | null;
  hi?: (number | null)[] | null;
  n?: number[] | null;
}

export interface Mark {
  kind: "vline" | "hline";
  at: string | number;
  label: string;
}

export interface FigureSpec {
  id: string;
  kind: Kind;
  title: string;
  x_label: string;
  y_label: string;
  series: Series[];
  marks: Mark[];
  nodes?: GraphNode[];
  edges?: GraphEdge[];
  moment?: "ready" | "run";
  note: string;
  draws_on: string[];
}

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

export const SIZE = { width: 560, height: 260 };
export const PAD = { left: 52, right: 12, top: 10, bottom: 46 };

export function plotBox(): Box {
  return { left: PAD.left, top: PAD.top, width: SIZE.width - PAD.left - PAD.right, height: SIZE.height - PAD.top - PAD.bottom };
}

// The x axis is categorical for bars and for any series whose x values are strings; numeric otherwise.
export function categorical(spec: FigureSpec): boolean {
  return spec.kind === "bars" || spec.series.some((s) => s.x.some((x) => typeof x === "string"));
}

export function categories(spec: FigureSpec): string[] {
  const out: string[] = [];
  for (const s of spec.series) for (const x of s.x) if (!out.includes(String(x))) out.push(String(x));
  return out;
}

export function extent(values: (number | null | undefined)[], includeZero = false): [number, number] {
  const v = values.filter((x): x is number => x !== null && x !== undefined && Number.isFinite(x));
  if (!v.length) return [0, 1];
  let lo = Math.min(...v);
  let hi = Math.max(...v);
  if (includeZero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  return [lo, hi];
}

export function yExtent(spec: FigureSpec): [number, number] {
  const ys: (number | null)[] = [];
  for (const s of spec.series) {
    ys.push(...s.y);
    if (s.lo) ys.push(...s.lo);
    if (s.hi) ys.push(...s.hi);
  }
  for (const m of spec.marks) if (m.kind === "hline" && typeof m.at === "number") ys.push(m.at);
  const [lo, hi] = extent(ys, spec.kind === "bars" || spec.kind === "density");
  const pad = (hi - lo) * 0.06;
  return [spec.kind === "bars" || spec.kind === "density" ? Math.min(lo, 0) : lo - pad, hi + pad];
}

export function xExtent(spec: FigureSpec): [number, number] {
  const xs: number[] = [];
  for (const s of spec.series) for (const x of s.x) if (typeof x === "number") xs.push(x);
  for (const m of spec.marks) if (m.kind === "vline" && typeof m.at === "number") xs.push(m.at);
  const [lo, hi] = extent(xs);
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}

export function scale([lo, hi]: [number, number], [a, b]: [number, number]): (v: number) => number {
  const k = (b - a) / (hi - lo || 1);
  return (v) => a + (v - lo) * k;
}

// "Nice" tick values: at most `n`, on 1, 2, or 5 steps.
export function ticks([lo, hi]: [number, number], n = 5): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? 10 * mag;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Number(t.toFixed(10)));
  return out;
}

export function fmtTick(v: number): string {
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  const s = v.toPrecision(3);
  return s.includes("e") ? s : s.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

export interface Bar {
  series: number;
  i: number;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  value: number;
}

// Grouped bars: one group per category, one bar per series inside it.
export function bars(spec: FigureSpec, box: Box, yDomain: [number, number]): Bar[] {
  const cats = categories(spec);
  const sy = scale(yDomain, [box.top + box.height, box.top]);
  const groupW = box.width / Math.max(cats.length, 1);
  const inner = groupW * 0.8;
  const barW = inner / Math.max(spec.series.length, 1);
  const zero = sy(Math.max(yDomain[0], 0));
  const out: Bar[] = [];
  spec.series.forEach((s, si) => {
    s.x.forEach((x, i) => {
      const v = s.y[i];
      if (v === null || v === undefined) return;
      const ci = cats.indexOf(String(x));
      const x0 = box.left + ci * groupW + (groupW - inner) / 2 + si * barW;
      const y0 = sy(v);
      out.push({ series: si, i, x: x0, y: Math.min(y0, zero), width: barW, height: Math.abs(zero - y0), label: String(x), value: v });
    });
  });
  return out;
}

export interface Pt {
  series: number;
  i: number;
  x: number;
  y: number;
  lo?: number;
  hi?: number;
  value: number;
  label: string;
}

// Points for lines, points, density, interval: x numeric or categorical (centred in the category slot).
export function points(spec: FigureSpec, box: Box, yDomain: [number, number]): Pt[] {
  const cat = categorical(spec);
  const cats = cat ? categories(spec) : [];
  const sxCat = (x: string | number) => box.left + (cats.indexOf(String(x)) + 0.5) * (box.width / Math.max(cats.length, 1));
  const sxNum = scale(xExtent(spec), [box.left, box.left + box.width]);
  const sy = scale(yDomain, [box.top + box.height, box.top]);
  const out: Pt[] = [];
  spec.series.forEach((s, si) => {
    s.x.forEach((x, i) => {
      const v = s.y[i];
      if (v === null || v === undefined) return;
      const p: Pt = { series: si, i, x: cat ? sxCat(x) : sxNum(Number(x)), y: sy(v), value: v, label: String(x) };
      const lo = s.lo?.[i];
      const hi = s.hi?.[i];
      if (lo !== null && lo !== undefined && hi !== null && hi !== undefined) {
        p.lo = sy(lo);
        p.hi = sy(hi);
      }
      out.push(p);
    });
  });
  return out;
}

export function path(pts: Pt[]): string {
  return pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
}

// Where a vertical mark sits on the x axis; null when the mark is off the axis or the axis is categorical without it.
export function markX(spec: FigureSpec, box: Box, m: Mark): number | null {
  if (m.kind !== "vline") return null;
  if (categorical(spec)) {
    const cats = categories(spec);
    const i = cats.indexOf(String(m.at));
    return i < 0 ? null : box.left + (i + 0.5) * (box.width / cats.length);
  }
  if (typeof m.at !== "number") return null;
  return scale(xExtent(spec), [box.left, box.left + box.width])(m.at);
}

export const NODE_R = 16;
export const LABEL_W = 96; // room one node label needs, in pixels

export interface LaidNode {
  id: string;
  label: string;
  role: Role;
  x: number;
  y: number;
  labelBelow?: boolean; // false: the label sits above the node, to stay clear of a neighbour's
}

export interface LaidEdge {
  i: number;
  src: string;
  dst: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

// A causal graph laid out by role: the instrument at the far left, the treatment left, the mediator between, the
// outcome right; what drives both above, spread between treatment and outcome; what the lane set aside below, dimmed.
// Arrows run centre to centre, shortened by the node radius at both ends.
export function graphLayout(spec: FigureSpec, box: Box): { nodes: LaidNode[]; edges: LaidEdge[] } {
  const nodes = spec.nodes ?? [];
  const yMid = box.top + box.height * 0.58;
  const tx = box.left + box.width * 0.2;
  const ox = box.left + box.width * 0.86;
  const fixed: Record<string, [number, number]> = { instrument: [box.left + box.width * 0.04, yMid], treatment: [tx, yMid], mediator: [(tx + ox) / 2, yMid], outcome: [ox, yMid] };
  const above = nodes.filter((n) => ["confounder", "driver", "hidden", "other"].includes(n.role));
  const below = nodes.filter((n) => n.role === "excluded");
  // The rows above and below use the whole width, so their labels have room; a row's labels alternate up and down when
  // its nodes sit closer than a label is wide.
  const spread = (list: GraphNode[], y: number): LaidNode[] => {
    const left = box.left + NODE_R + 4;
    const span = box.width - 2 * NODE_R - 8;
    const step = span / Math.max(list.length, 1);
    return list.map((n, i) => ({ ...n, x: left + (i + 0.5) * step, y, labelBelow: step >= LABEL_W || i % 2 === 0 }));
  };
  const laid: LaidNode[] = [];
  const seen = new Set<string>();
  for (const n of nodes) {
    const f = fixed[n.role];
    if (f && !seen.has(n.role)) {
      laid.push({ ...n, x: f[0], y: f[1] });
      seen.add(n.role);
    }
  }
  const rest = nodes.filter((n) => !laid.some((l) => l.id === n.id) && n.role !== "excluded");
  laid.push(...spread(rest.length === above.length ? above : rest, box.top + NODE_R + 6));
  laid.push(...spread(below, box.top + box.height - NODE_R - 2));
  const at = (id: string) => laid.find((n) => n.id === id);
  const edges: LaidEdge[] = [];
  (spec.edges ?? []).forEach((e, i) => {
    const a = at(e.src);
    const b = at(e.dst);
    if (!a || !b) return;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const d = Math.hypot(dx, dy) || 1;
    const ux = dx / d;
    const uy = dy / d;
    edges.push({ i, src: e.src, dst: e.dst, x1: a.x + ux * NODE_R, y1: a.y + uy * NODE_R, x2: b.x - ux * (NODE_R + 3), y2: b.y - uy * (NODE_R + 3) });
  });
  return { nodes: laid, edges };
}

// Category axis labels: slanted when the slots are narrower than a label, and cut to what a slot can hold.
export function categoryLabels(cats: string[], box: Box): { text: string; slant: boolean }[] {
  const slot = box.width / Math.max(cats.length, 1);
  const slant = slot < 72;
  const max = slant ? 22 : Math.max(6, Math.floor(slot / 6.5));
  return cats.map((c) => ({ text: c.length > max ? `${c.slice(0, max - 1)}…` : c, slant }));
}
