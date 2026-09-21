import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { FigureSpec } from "../figure";
import { series } from "../figure";
import Figure from "./Figure";

const spec: FigureSpec = {
  id: "overlap_lunch",
  kind: "bars",
  nodes: [],
  edges: [],
  moment: "run",
  title: "Who got the change, by lunch",
  x_label: "level",
  y_label: "share of the arm",
  note: "both arms appear at every level",
  draws_on: ["probe:adjustment.overlap"],
  series: [
    series({ name: "test preparation course = completed", x: ["lunch = free/reduced", "lunch = standard"], y: [0.3, 0.7], n: [107, 251] }),
    series({ name: "test preparation course ≠ completed", x: ["lunch = free/reduced", "lunch = standard"], y: [0.4, 0.6], n: [248, 394] }),
  ],
  marks: [],
};

describe("Figure", () => {
  it("draws one bar per value, each carrying its address", () => {
    const html = renderToStaticMarkup(createElement(Figure, { spec }));
    expect((html.match(/<rect/g) ?? []).length).toBe(4);
    expect(html).toContain("[figure:overlap_lunch.test_preparation_course_completed.1] test preparation course = completed · lunch = standard: 0.7 (n=251)");
    expect(html).toContain("figure:overlap_lunch");
    expect(html).toContain("both arms appear at every level");
  });
  it("draws lines with a mark for the change", () => {
    const lines: FigureSpec = {
      ...spec,
      id: "by_group",
      kind: "lines",
      nodes: [],
      edges: [],
      moment: "run",
      marks: [{ kind: "vline", at: 2002, label: "the change" }],
      series: [
        series({ name: "got the change", x: [2000, 2001, 2002, 2003], y: [1, 2, 3, 7] }),
        series({ name: "did not", x: [2000, 2001, 2002, 2003], y: [1, 2, 3, 4] }),
      ],
    };
    const html = renderToStaticMarkup(createElement(Figure, { spec: lines }));
    expect((html.match(/<path/g) ?? []).length).toBe(2);
    expect((html.match(/<circle/g) ?? []).length).toBe(8);
    expect(html).toContain("the change");
  });
});

describe("Figure graph", () => {
  it("draws a circle per node and an arrow per edge, each carrying its address", () => {
    const graph: FigureSpec = {
      id: "causal_graph",
      kind: "graph",
      moment: "run",
      title: "What the lane drew",
      x_label: "",
      y_label: "",
      note: "lunch drives both",
      draws_on: ["design.graph"],
      series: [],
      marks: [],
      nodes: [
        { id: "course", label: "course", role: "treatment" },
        { id: "math", label: "math score", role: "outcome" },
        { id: "lunch", label: "lunch", role: "confounder" },
        { id: "u", label: "unobserved", role: "hidden" },
      ],
      edges: [
        { src: "course", dst: "math", cites: [] },
        { src: "lunch", dst: "course", cites: ["col:lunch.when"] },
        { src: "lunch", dst: "math", cites: [] },
      ],
    };
    const html = renderToStaticMarkup(createElement(Figure, { spec: graph }));
    expect((html.match(/<circle/g) ?? []).length).toBe(4);
    expect((html.match(/marker-end="url\(#arrow\)"/g) ?? []).length).toBe(3);
    expect(html).toContain("[figure:causal_graph.edge.1] lunch → course (col:lunch.when)");
    expect(html).toContain("[figure:causal_graph.node.3] unobserved (hidden)");
    expect(html).toContain('stroke-dasharray="3 3"');
    expect(html).toContain("lunch drives both");
  });
});
