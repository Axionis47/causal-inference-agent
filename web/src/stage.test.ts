import { describe, expect, it } from "vitest";
import { stageLabel } from "./stage";
import type { DatasetSummary } from "./types";

const d = (over: Partial<DatasetSummary>): DatasetSummary => ({
  name: "x",
  title: "X",
  csv: "x.csv",
  rows: null,
  columns: null,
  created_at: null,
  shipped: false,
  has_claims: false,
  question: null,
  session: null,
  ...over,
});

describe("stageLabel", () => {
  it("covers every branch", () => {
    expect(stageLabel(d({ shipped: true }))).toEqual({ text: "shipped", tone: "" });
    expect(stageLabel(d({}))).toEqual({ text: "not started", tone: "" });
    expect(stageLabel(d({ session: { stage: "busy", phase: "before", runs: 0 } }))).toEqual({ text: "working", tone: "amber" });
    expect(stageLabel(d({ session: { stage: "ended", phase: "after", runs: 2 } }))).toEqual({ text: "ended · 2 runs", tone: "" });
    expect(stageLabel(d({ session: { stage: "ended", phase: "before", runs: 0 } }))).toEqual({ text: "ended", tone: "" });
    expect(stageLabel(d({ session: { stage: "error", phase: "before", runs: 0 } }))).toEqual({ text: "error", tone: "strike" });
    expect(stageLabel(d({ session: { stage: "waiting", phase: "after", runs: 1 } }))).toEqual({ text: "1 run", tone: "teal" });
    expect(stageLabel(d({ session: { stage: "waiting", phase: "before", runs: 0 } }))).toEqual({ text: "interviewing", tone: "amber" });
  });
});
