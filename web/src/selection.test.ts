import { describe, expect, it } from "vitest";
import { parseSelection, resolveSelection, serialiseSelection } from "./selection";
import type { RunView } from "./types";

const run = (index: number, files: string[] = []): RunView =>
  ({ index, question: "", family: null, specialist: null, status: "ok", run_id: `r${index}`, effect: null, ci_low: null, ci_high: null, estimator: null, decision: {}, decision_record: "", flags: [], checks: [], refutations: [], interpretations: [], estimates: [], feasibility: null, files, what_if: {}, differs: [], figures: [] }) as RunView;

describe("selection hash", () => {
  it("round-trips every form", () => {
    for (const h of ["", "#claims", "#runs", "#runs/2", "#files", "#files/2", "#files/2/report.md", "#files/1/a%20b.csv"]) {
      expect(serialiseSelection(parseSelection(h))).toBe(h);
    }
  });
  it("rejects anything else", () => {
    for (const h of ["#x", "#claims/1", "#runs/a", "#runs/1/x", "#files/1/", "#files/1/a/b", "#files/%E0"]) expect(parseSelection(h)).toBeNull();
  });
  it("keeps dots and encoded characters in file names", () => {
    expect(parseSelection("#files/3/design.json")).toEqual({ tab: "files", run: 3, file: "design.json" });
    expect(serialiseSelection({ tab: "files", run: 3, file: "a b.csv" })).toBe("#files/3/a%20b.csv");
  });
});

describe("resolveSelection", () => {
  const runs = [run(1, ["report.md"]), run(2, ["artifacts.json"])];
  it("leaves claims and closed alone", () => {
    expect(resolveSelection(null, runs)).toBeNull();
    expect(resolveSelection({ tab: "claims" }, runs)).toEqual({ tab: "claims" });
  });
  it("falls back to the last run", () => {
    expect(resolveSelection({ tab: "runs" }, runs)).toEqual({ tab: "runs", run: 2 });
    expect(resolveSelection({ tab: "runs", run: 9 }, runs)).toEqual({ tab: "runs", run: 2 });
    expect(resolveSelection({ tab: "runs", run: 1 }, runs)).toEqual({ tab: "runs", run: 1 });
  });
  it("drops a file the run does not have", () => {
    expect(resolveSelection({ tab: "files", run: 1, file: "x.csv" }, runs)).toEqual({ tab: "files", run: 1 });
    expect(resolveSelection({ tab: "files", run: 1, file: "report.md" }, runs)).toEqual({ tab: "files", run: 1, file: "report.md" });
  });
  it("keeps the tab when there are no runs", () => {
    expect(resolveSelection({ tab: "files", run: 1, file: "report.md" }, [])).toEqual({ tab: "files" });
  });
});
