import { describe, expect, it } from "vitest";
import { cell, ci, count, num } from "./fmt";

describe("fmt", () => {
  it("num keeps four significant figures and trims zeros", () => {
    expect(num(null)).toBe("—");
    expect(num(0)).toBe("0");
    expect(num(1000)).toBe("1000");
    expect(num(2.5)).toBe("2.5");
    expect(num(0.123456)).toBe("0.1235");
    expect(num(-0.00001234)).toBe("-0.00001234");
  });
  it("ci and count", () => {
    expect(ci(null, null)).toBe("[—, —]");
    expect(ci(-1.5, 2)).toBe("[-1.5, 2]");
    expect(count(12345)).toBe((12345).toLocaleString());
    expect(count(null)).toBe("—");
  });
  it("cell handles every value kind", () => {
    expect(cell(undefined)).toBe("—");
    expect(cell("")).toBe("—");
    expect(cell(true)).toBe("yes");
    expect(cell(0.05)).toBe("0.05");
    expect(cell("ok")).toBe("ok");
    expect(cell({ a: 1 })).toBe('{"a":1}');
  });
});
