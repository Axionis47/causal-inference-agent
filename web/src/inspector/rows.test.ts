import { describe, expect, it } from "vitest";
import { checkRows, decisionOver, decisionWhy, declineRows, estimateRows, falsificationRows } from "./rows";
import type { RunView } from "../types";

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
    falsifications: [],
    sensitivities: [],
    lead_selection: {},
    lead_design_id: "d1",
    context_id: "context",
    context_admission: null,
    context_deficiency: null,
    terminal: {},
    candidate_history: [],
    reasoning_history: [],
    context_facts: {},
    designs: {},
    ledger: {},
    candidates: [],
    episode_visits: {},
    comparisons: [],
    execution_attempts: [],
    interpretations: [],
    estimates: [],
    feasibility: null,
    files: [],
    ...over,
  }) as RunView;

describe("run rows", () => {
  it("puts the primary estimate first and keeps flags", () => {
    const r = run({
      estimates: [
        {
          design_id: "d1",
          endpoint_id: "total.backdoor",
          target: {},
          confidence_set: null,
          modifier: null,
          level: null,
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
          design_id: "d1",
          endpoint_id: "total.backdoor",
          target: {},
          confidence_set: null,
          modifier: null,
          level: null,
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
          design_id: "d1",
          endpoint_id: "total.backdoor",
          target: {},
          confidence_set: null,
          modifier: null,
          level: null,
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
    expect(checkRows([{ design_id: "d1", contrast: null, name: "overlap", level: "warn", value: 0.02, threshold: 0.05, detail: null }])).toEqual([
      { contrast: "—", name: "overlap", level: "warn", value: "0.02", threshold: "0.05", detail: "" },
    ]);
    const rows = falsificationRows(
      run({
        falsifications: [
          { design_id: "d1", contrast: "all", refuter: "placebo", passed: null, p_value: null, new_effect: null, detail: "" },
          { design_id: "d1", contrast: "a", refuter: "subset", passed: false, p_value: 0.001, new_effect: 3, detail: "moved" },
        ],
      }),
    );
    expect(rows[0].passed).toBe("n/a");
    expect(rows[1]).toMatchObject({ passed: "failed", p_value: "0.001", new_effect: "3", kind: "d1" });
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
