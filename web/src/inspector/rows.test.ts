import { describe, expect, it } from "vitest";
import { checkRows, claimRows, decisionOver, decisionWhy, declineRows, estimateRows, refutationRows, statusMatrix } from "./rows";
import type { ClaimView, RunView, StatusView } from "../types";

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
        { contrast: null, method: "ols", value: 1.5, ci_low: 1, ci_high: 2, n: 10, n_treated: 4, n_control: 6, secondary: true, error: null },
        { contrast: "a", method: "ipw", value: 2, ci_low: 1.5, ci_high: 2.5, n: 10, n_treated: 4, n_control: 6, secondary: false, error: null },
        { contrast: "a", method: "dml", value: null, ci_low: null, ci_high: null, n: null, n_treated: null, n_control: null, secondary: false, error: "boom" },
      ],
    });
    const rows = estimateRows(r);
    expect(rows.map((x) => x.method)).toEqual(["ipw", "ols", "dml"]);
    expect(rows[0]).toMatchObject({ primary: true, value: "2", ci: "[1.5, 2.5]", contrast: "a" });
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
