// A figure is data with addresses (see causal_agent/viz/spec.py). This module turns a spec into the numbers an SVG
// needs: scales, ticks, bar and point positions. Pure functions, tested; Figure.tsx only draws what comes out.

export type Kind = "bars" | "lines" | "points" | "density" | "interval";

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
