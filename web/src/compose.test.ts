import { describe, expect, it } from "vitest";
import { composeAnswer } from "./compose";

describe("composeAnswer", () => {
  it("quotes each answered question beside its answer, then the free text", () => {
    const out = composeAnswer("and the scores are out of 100", [
      { question: "I read grain from your note.  Is that right?", answer: "Yes, that's right" },
      { question: "How were rows chosen?", answer: "whole" },
    ]);
    expect(out).toBe(
      'On "I read grain from your note. Is that right?": Yes, that\'s right.\nOn "How were rows chosen?": whole.\nand the scores are out of 100',
    );
  });
  it("passes free text alone through unchanged and drops empty answers", () => {
    expect(composeAnswer("  run  ", [])).toBe("run");
    expect(composeAnswer("", [{ question: "q", answer: "  " }])).toBe("");
  });
});
