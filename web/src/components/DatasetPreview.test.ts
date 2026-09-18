import { describe, expect, it } from "vitest";
import { columnWarnings, datasetWarnings, grainText } from "./DatasetPreview";
import type { ColumnSummary, ProfileOut } from "../types";

const col = (over: Partial<ColumnSummary> = {}): ColumnSummary => ({
  name: "x", key: "x", kind: "numeric", nulls: 0, null_rate: 0, distinct: 3, constant: false, examples: [],
  numeric: null, top_values: [], datetime: null, sentinels: [], issues: [], ...over,
});

const prof = (over: Partial<ProfileOut> = {}): ProfileOut => ({
  upload_id: "0000abcd", filename: "t.csv", rows: 3, columns: [col()], head: [], duplicate_rows: 0, candidate_keys: [], grain: null, co_missing: [], issues: [], ...over,
});

describe("grainText", () => {
  it("names the grain when the profiler found one", () => {
    expect(grainText(prof({ grain: ["id"] }))).toBe("One row per id.");
    expect(grainText(prof({ grain: ["state", "year"] }))).toBe("One row per state + year.");
  });
  it("falls back to the first candidate key, then says there is none", () => {
    expect(grainText(prof({ candidate_keys: [["a", "b"]] }))).toBe("One row per a + b.");
    expect(grainText(prof())).toBe("No column, or pair of columns, identifies a row.");
  });
});

describe("warnings", () => {
  it("collects the dataset's facts as short lines", () => {
    expect(datasetWarnings(prof({ duplicate_rows: 1, co_missing: [["a", "b"]], issues: ["one or more headers have leading or trailing whitespace"] }))).toEqual([
      "one or more headers have leading or trailing whitespace",
      "1 duplicate row",
      "a and b are missing on the same rows",
    ]);
    expect(datasetWarnings(prof())).toEqual([]);
  });
  it("collects a column's oddities", () => {
    expect(columnWarnings(col({ constant: true, sentinels: [{ value: "-999", count: 4, reason: "placeholder number at the edge of the range" }], issues: ["values have leading or trailing whitespace"] }))).toEqual([
      "values have leading or trailing whitespace",
      "every row has the same value",
      "-999 × 4: placeholder number at the edge of the range",
    ]);
  });
});
