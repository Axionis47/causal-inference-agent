import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { FigureSpec } from "../figure";
import Figure from "./Figure";

const spec: FigureSpec = {
  id: "overlap_lunch", kind: "bars", title: "Who got the change, by lunch", x_label: "level", y_label: "share of the arm", note: "both arms appear at every level", draws_on: ["probe:adjustment.overlap"],
  series: [
    { name: "test preparation course = completed", x: ["lunch = free/reduced", "lunch = standard"], y: [0.3, 0.7], n: [107, 251] },
    { name: "test preparation course ≠ completed", x: ["lunch = free/reduced", "lunch = standard"], y: [0.4, 0.6], n: [248, 394] },
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
    const lines: FigureSpec = { ...spec, id: "by_group", kind: "lines", marks: [{ kind: "vline", at: 2002, label: "the change" }],
      series: [{ name: "got the change", x: [2000, 2001, 2002, 2003], y: [1, 2, 3, 7] }, { name: "did not", x: [2000, 2001, 2002, 2003], y: [1, 2, 3, 4] }] };
    const html = renderToStaticMarkup(createElement(Figure, { spec: lines }));
    expect((html.match(/<path/g) ?? []).length).toBe(2);
    expect((html.match(/<circle/g) ?? []).length).toBe(8);
    expect(html).toContain("the change");
  });
});
