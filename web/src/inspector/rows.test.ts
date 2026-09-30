import { describe, expect, it } from "vitest";
import { checkRows, claimRows, decisionOver, decisionWhy, declineRows, estimateRows, journalGroups, refutationRows, statusMatrix } from "./rows";
import type { ClaimView, RunView, StatusView, StepView } from "../types";

const claim = (over: Partial<ClaimView>): ClaimView => ({
  key: "k",
  kind: "unit",
  status: "drafted",
  fields: {},
  source: null,
  evidence: [],
  check_detail: null,
  asked: 0,
  refutations: 0,
  ...over,
});
const run = (over: Partial<RunView>): RunView =>
  ({
    index: 1,
    question: "",
    family: null,
    specialist: null,
    status: "ok",
    run_id: "r1",
    effect: null,
    ci_low: null,
    ci_high: null,
    estimator: null,
    decision: { chosen: null, chosen_assumption: null, why: null, over: {} },
    decision_record: "",
    flags: [],
    checks: [],
    refutations: [],
    interpretations: [],
    estimates: [],
    feasibility: null,
    files: [],
    ...over,
  }) as RunView;

describe("claimRows", () => {
  it("drops empty fields and joins arrays", () => {
    const [r] = claimRows([claim({ fields: { a: "x", b: null, c: "", d: [], e: [1, 2], f: { z: 1 } } })]);
    expect(r.fields).toEqual([
      { k: "a", v: "x" },
      { k: "e", v: "1, 2" },
      { k: "f", v: '{"z":1}' },
    ]);
  });
});

describe("statusMatrix", () => {
  const status: StatusView = {
    table: { did: { unit: "fits", time: "unknown" }, rd: { unit: "does_not_fit" } },
    surviving: ["did"],
    struck: { rd: "no cutoff" },
    required: [],
    settled: [],
    open: [],
    ready: false,
    contradictions: [],
  };
  it("orders kinds by first appearance in claims, then the table's own", () => {
    const m = statusMatrix(status, [claim({ kind: "time" }), claim({ kind: "unit" })])!;
    expect(m.rows.map((r) => r.kind)).toEqual(["time", "unit"]);
    expect(m.families).toEqual([
      { name: "did", struck: null },
      { name: "rd", struck: "no cutoff" },
    ]);
    expect(m.rows[0].cells).toEqual({ did: "unknown", rd: "not_needed" });
  });
  it("is null without a table", () => {
    expect(statusMatrix(null, [])).toBeNull();
    expect(statusMatrix({ ...status, table: {} }, [])).toBeNull();
  });
});

describe("run rows", () => {
  it("puts the primary estimate first and keeps flags", () => {
    const r = run({
      estimates: [
        {
          contrast: null,
          method: "ols",
          value: 1.5,
          ci_low: 1,
          ci_high: 2,
          n: 10,
          n_treated: 4,
          n_control: 6,
          secondary: true,
          error: null,
          p_value: null,
          p_value_source: null,
        },
        {
          contrast: "a",
          method: "ipw",
          value: 2,
          ci_low: 1.5,
          ci_high: 2.5,
          n: 10,
          n_treated: 4,
          n_control: 6,
          secondary: false,
          error: null,
          p_value: 0.03,
          p_value_source: "ritest",
        },
        {
          contrast: "a",
          method: "dml",
          value: null,
          ci_low: null,
          ci_high: null,
          n: null,
          n_treated: null,
          n_control: null,
          secondary: false,
          error: "boom",
          p_value: null,
          p_value_source: null,
        },
      ],
    });
    const rows = estimateRows(r);
    expect(rows.map((x) => x.method)).toEqual(["ipw", "ols", "dml"]);
    expect(rows[0]).toMatchObject({ primary: true, value: "2", ci: "[1.5, 2.5]", contrast: "a", p: "0.03 (ritest)" });
    expect(rows[1]).toMatchObject({ p: "—" });
    expect(rows[2]).toMatchObject({ error: "boom", value: "—" });
  });
  it("shapes checks and refutations", () => {
    expect(checkRows([{ contrast: null, name: "overlap", level: "warn", value: 0.02, threshold: 0.05, detail: null }])).toEqual([
      { contrast: "—", name: "overlap", level: "warn", value: "0.02", threshold: "0.05", detail: "" },
    ]);
    const rows = refutationRows(
      run({
        refutations: [
          { contrast: null, refuter: "placebo", kind: "placebo_treatment", passed: null, p_value: null, new_effect: null, detail: null },
          { contrast: "a", refuter: "subset", kind: null, passed: false, p_value: 0.001, new_effect: 3, detail: "moved" },
        ],
      }),
    );
    expect(rows[0].passed).toBe("n/a");
    expect(rows[1]).toMatchObject({ passed: "failed", p_value: "0.001", new_effect: "3", kind: "—" });
  });
  it("reads the decision bag", () => {
    const r = run({ decision: { chosen: null, chosen_assumption: null, why: "fits", over: { rd: "no cutoff", iv: "no instrument" } } });
    expect(decisionWhy(r)).toBe("fits");
    expect(decisionOver(r)).toEqual([
      { family: "rd", reason: "no cutoff" },
      { family: "iv", reason: "no instrument" },
    ]);
    expect(decisionWhy(run({}))).toBeNull();
    expect(decisionOver(run({ decision: { over: "x" } as unknown as RunView["decision"] }))).toEqual([]);
  });
});

describe("declineRows", () => {
  it("keeps the lane's order and shows a dash for what was not given", () => {
    const r = run({
      declines: [
        {
          address: "decline:load.scope_window",
          stage: "load",
          kind: "declined",
          about: "scope.window",
          pack_value: "the spring term",
          took: null,
          reason: "not in a form the code can apply",
          check: "intake.window_unparsed",
        },
        {
          address: "decline:load.scope_target",
          stage: "load",
          kind: "substituted",
          about: "scope.target",
          pack_value: "on_treated",
          took: "effect_at_cutoff",
          reason: "no average over the treated",
          check: "target.effect_at_cutoff",
        },
      ],
    });
    const rows = declineRows(r);
    expect(rows.map((x) => x.about)).toEqual(["scope.window", "scope.target"]);
    expect(rows[0].took).toBe("—");
    expect(rows[1]).toMatchObject({ kind: "substituted", packValue: "on_treated", took: "effect_at_cutoff", check: "target.effect_at_cutoff" });
    expect(declineRows(run({}))).toEqual([]);
  });
});

const step = (over: Partial<StepView>): StepView => ({
  n: 1,
  address: "step:1",
  kind: "claim",
  by: "person",
  at: "2026-09-27T00:00:00+00:00",
  memory_version: 1,
  design: null,
  read: [],
  left: [],
  note: "",
  run: null,
  run_id: null,
  ...over,
});

describe("journalGroups", () => {
  it("groups by design run: the steps before a design belong to it, a what-if after run 1 stays with run 1, design 2 starts group 2", () => {
    const steps = [
      step({ n: 1, address: "step:1", kind: "question", by: "model" }),
      step({ n: 2, address: "step:2", kind: "claim" }),
      step({ n: 3, address: "step:3", kind: "design", by: "code", design: 1, left: ["designs/1"], run: 1 }),
      step({ n: 4, address: "step:4", kind: "run", by: "code", design: 1, left: ["/runs/x-1", "designs/1/record.json"], run: 1, run_id: "x-1" }),
      step({ n: 5, address: "step:5", kind: "brief", by: "code", design: 1, run: 1 }),
      step({ n: 6, address: "step:6", kind: "what_if", design: 1 }),
      step({ n: 7, address: "step:7", kind: "design", by: "code", design: 2, left: ["designs/2"], run: 2 }),
      step({ n: 8, address: "step:8", kind: "run", by: "code", design: 2, run: 2 }),
      step({ n: 9, address: "step:9", kind: "requestion", design: 2 }),
      step({ n: 10, address: "step:10", kind: "question", by: "model" }),
    ];
    const groups = journalGroups(steps);
    expect(groups.map((g) => [g.design, g.rows.map((r) => r.n)])).toEqual([
      [1, [1, 2, 3, 4, 5, 6]],
      [2, [7, 8, 9]],
      [null, [10]],
    ]);
    const run = groups[0].rows[3];
    expect(run.run).toBe(1);
    expect(run.hasFiles).toBe(true);
    expect(groups[0].rows[2].hasFiles).toBe(false);
  });
  it("is empty for no steps", () => {
    expect(journalGroups([])).toEqual([]);
  });
});
