import { describe, expect, it } from "vitest";
import { clampPaneWidth, parsePaneWidth, parseSidebar } from "./layout";

describe("layout prefs", () => {
  it("sidebar defaults to open", () => {
    expect(parseSidebar(null)).toBe(true);
    expect(parseSidebar("junk")).toBe(true);
    expect(parseSidebar("closed")).toBe(false);
  });
  it("pane width parses only positive numbers", () => {
    expect(parsePaneWidth(null)).toBeNull();
    expect(parsePaneWidth("abc")).toBeNull();
    expect(parsePaneWidth("-5")).toBeNull();
    expect(parsePaneWidth("512.6")).toBe(513);
  });
  it("clamps to [360, 70%]", () => {
    expect(clampPaneWidth(100, 1000)).toBe(360);
    expect(clampPaneWidth(900, 1000)).toBe(700);
    expect(clampPaneWidth(500, 1000)).toBe(500);
  });
  it("never lets the minimum exceed the maximum on a narrow desk", () => {
    expect(clampPaneWidth(1000, 400)).toBe(280);
    expect(clampPaneWidth(10, 400)).toBe(280);
  });
});
