import { describe, expect, it } from "vitest";
import { parseCsv } from "./table";

describe("parseCsv", () => {
  it("reads the header and the first rows, counting the rest", () => {
    const t = parseCsv('lunch,n\nstandard,645\n"free/reduced",355\nx,1\n', 2);
    expect(t.header).toEqual(["lunch", "n"]);
    expect(t.rows).toEqual([
      ["standard", "645"],
      ["free/reduced", "355"],
    ]);
    expect(t.total).toBe(3);
  });
  it("keeps a quoted comma and a doubled quote", () => {
    expect(parseCsv('a,b\n"x, y","say ""hi"""\n').rows).toEqual([["x, y", 'say "hi"']]);
  });
  it("is empty on nothing", () => {
    expect(parseCsv("")).toEqual({ header: [], rows: [], total: 0 });
  });
});

it("keeps quoted newlines inside the original table row", () => {
  expect(parseCsv('name,note\r\nA,"first\nsecond"\r\nB,"a,b"\r\n')).toEqual({
    header: ["name", "note"],
    rows: [
      ["A", "first\nsecond"],
      ["B", "a,b"],
    ],
    total: 2,
  });
});
