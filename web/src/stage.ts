import type { DatasetSummary } from "./types";

export type Tone = "" | "teal" | "amber" | "strike";

const runs = (n: number) => `${n} run${n === 1 ? "" : "s"}`;

/** The one word (or two) that says where a dataset's conversation stands. */
export function stageLabel(d: DatasetSummary): { text: string; tone: Tone } {
  if (d.shipped) return { text: "shipped", tone: "" };
  const s = d.session;
  if (!s) return { text: "not started", tone: "" };
  if (s.stage === "busy") return { text: "working", tone: "amber" };
  if (s.stage === "ended") return { text: s.runs ? `ended · ${runs(s.runs)}` : "ended", tone: "" };
  if (s.stage === "error") return { text: "error", tone: "strike" };
  if (s.phase === "after") return { text: runs(s.runs), tone: "teal" };
  return { text: "interviewing", tone: "amber" };
}
