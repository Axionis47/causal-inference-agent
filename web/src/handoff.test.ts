import { describe, expect, it } from "vitest";
import { fieldGroups, valueText } from "./handoff";
import type { HandoffField } from "./types";

const f = (address: string, value: unknown = "x"): HandoffField => ({ address, value, status: "confirmed", source: null, said: null, reason: null });

describe("fieldGroups", () => {
  it("puts the claims first and one group per column, in order", () => {
    const g = fieldGroups([f("claim:change.what"), f("claim:grain.panel", false), f("col:lunch.when"), f("col:lunch.meaning"), f("col:math_score.when")]);
    expect(g.map((x) => x.title)).toEqual(["claims", "col:lunch", "col:math_score"]);
    expect(g[1].fields.map((x) => x.address)).toEqual(["col:lunch.when", "col:lunch.meaning"]);
  });
});

describe("valueText", () => {
  it("renders every value as text the person can edit back", () => {
    expect(valueText(null)).toBe("");
    expect(valueText(["lunch", "gender"])).toBe("lunch, gender");
    expect(valueText(false)).toBe("false");
    expect(valueText(5.5)).toBe("5.5");
  });
});
