import { describe, expect, it } from "vitest";
import { bars, categorical, categories, extent, fmtTick, graphLayout, markX, path, plotBox, points, ticks, yExtent, type FigureSpec } from "./figure";

const barSpec: FigureSpec = {
  id: "overlap_lunch", kind: "bars", title: "t", x_label: "", y_label: "", note: "", draws_on: [],
  series: [
    { name: "completed", x: ["lunch = free", "lunch = standard"], y: [0.3, 0.7], n: [100, 250] },
    { name: "none", x: ["lunch = free", "lunch = standard"], y: [0.4, 0.6], n: [260, 390] },
  ],
  marks: [],
};

const lineSpec: FigureSpec = {
  id: "by_group", kind: "lines", title: "t", x_label: "year", y_label: "y", note: "", draws_on: [],
  series: [
    { name: "got the change", x: [2000, 2001, 2002, 2003], y: [1, 2, 3, 7] },
    { name: "did not", x: [2000, 2001, 2002, 2003], y: [1, 2, 3, 4] },
  ],
  marks: [{ kind: "vline", at: 2002, label: "the change" }],
};

describe("scales and ticks", () => {
  it("extent pads a flat range and can include zero", () => {
    expect(extent([5, 5])).toEqual([4, 6]);
    expect(extent([3, 9], true)).toEqual([0, 9]);
    expect(extent([null, undefined])).toEqual([0, 1]);
  });
  it("ticks land on 1-2-5 steps", () => {
    expect(ticks([0, 1])).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(ticks([0, 100], 5)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(fmtTick(0.2)).toBe("0.2");
    expect(fmtTick(12345)).toMatch(/12,?345/);
  });
  it("bars start at zero; lines pad the y range", () => {
    expect(yExtent(barSpec)[0]).toBe(0);
    const [lo, hi] = yExtent(lineSpec);
    expect(lo).toBeLessThan(1);
    expect(hi).toBeGreaterThan(7);
  });
});

describe("bars", () => {
  it("groups one bar per series inside each category, all above the axis", () => {
    const box = plotBox();
    const bs = bars(barSpec, box, yExtent(barSpec));
    expect(bs).toHaveLength(4);
    expect(categorical(barSpec)).toBe(true);
    expect(categories(barSpec)).toEqual(["lunch = free", "lunch = standard"]);
    const free = bs.filter((b) => b.label === "lunch = free");
    expect(free[0].x).toBeLessThan(free[1].x);
    expect(free[1].x - free[0].x).toBeCloseTo(free[0].width, 5);
    for (const b of bs) {
      expect(b.height).toBeGreaterThan(0);
      expect(b.y + b.height).toBeCloseTo(box.top + box.height, 5);
    }
    const tallest = bs.reduce((a, b) => (b.value > a.value ? b : a));
    expect(tallest.value).toBe(0.7);
    expect(tallest.height).toBeGreaterThan(free[0].height);
  });
});

describe("points and marks", () => {
  it("places numeric x on a linear scale and skips empty values", () => {
    const box = plotBox();
    const spec = { ...lineSpec, series: [{ ...lineSpec.series[0], y: [1, null, 3, 7] }] };
    const ps = points(spec, box, yExtent(spec));
    expect(ps.map((p) => p.i)).toEqual([0, 2, 3]);
    expect(ps[0].x).toBeLessThan(ps[1].x);
    expect(path(ps).startsWith("M")).toBe(true);
    expect(path(ps).split("L")).toHaveLength(3);
  });
  it("a vertical mark sits between the periods it separates", () => {
    const box = plotBox();
    const ps = points(lineSpec, box, yExtent(lineSpec));
    const x = markX(lineSpec, box, lineSpec.marks[0])!;
    expect(x).toBeCloseTo(ps[2].x, 5);
    expect(markX(barSpec, box, { kind: "vline", at: "nope", label: "" })).toBeNull();
    expect(markX(barSpec, box, { kind: "vline", at: "lunch = standard", label: "" })).toBeGreaterThan(box.left);
  });
  it("intervals carry lo and hi in pixels", () => {
    const spec: FigureSpec = { ...lineSpec, kind: "interval", series: [{ name: "e", x: [1, 2], y: [5, 6], lo: [4, 5], hi: [6, 7] }], marks: [] };
    const ps = points(spec, plotBox(), yExtent(spec));
    expect(ps[0].lo).toBeGreaterThan(ps[0].y);
    expect(ps[0].hi).toBeLessThan(ps[0].y);
  });
});

describe("graph layout", () => {
  const graph: FigureSpec = {
    id: "causal_graph", kind: "graph", title: "g", x_label: "", y_label: "", note: "", draws_on: ["design.graph"], series: [], marks: [],
    nodes: [
      { id: "course", label: "test preparation course", role: "treatment" },
      { id: "math", label: "math score", role: "outcome" },
      { id: "lunch", label: "lunch", role: "confounder" },
      { id: "gender", label: "gender", role: "driver" },
      { id: "reading", label: "reading score", role: "excluded" },
    ],
    edges: [{ src: "course", dst: "math", cites: [] }, { src: "lunch", dst: "course", cites: ["col:lunch.when"] }, { src: "lunch", dst: "math", cites: [] }, { src: "nope", dst: "math", cites: [] }],
  };
  it("puts the treatment left of the outcome, drivers above, excluded below, and drops an arrow to a missing node", () => {
    const box = plotBox();
    const { nodes, edges } = graphLayout(graph, box);
    const at = (id: string) => nodes.find((n) => n.id === id)!;
    expect(at("course").x).toBeLessThan(at("math").x);
    expect(at("course").y).toBeCloseTo(at("math").y, 5);
    expect(at("lunch").y).toBeLessThan(at("course").y);
    expect(at("gender").x).toBeGreaterThan(at("lunch").x);
    expect(at("reading").y).toBeGreaterThan(at("course").y);
    expect(edges.map((e) => e.i)).toEqual([0, 1, 2]);
    const e0 = edges[0];
    expect(e0.x1).toBeGreaterThan(at("course").x);
    expect(e0.x2).toBeLessThan(at("math").x);
  });
});
