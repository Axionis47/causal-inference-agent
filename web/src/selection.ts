// What the inspector shows, kept in the URL hash: "", #claims, #runs, #runs/2, #files, #files/2, #files/2/report.md.
import type { RunView } from "./types";

export type Tab = "claims" | "runs" | "files";
export type Selection = { tab: "claims" } | { tab: "runs"; run?: number } | { tab: "files"; run?: number; file?: string } | null;

export const TABS: Tab[] = ["claims", "runs", "files"];

export function parseSelection(hash: string): Selection {
  const h = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!h) return null;
  const parts = h.split("/");
  const tab = parts[0];
  if (tab === "claims") return parts.length === 1 ? { tab } : null;
  if (tab !== "runs" && tab !== "files") return null;
  if (parts.length === 1) return { tab };
  if (!/^\d+$/.test(parts[1])) return null;
  const run = Number(parts[1]);
  if (tab === "runs") return parts.length === 2 ? { tab, run } : null;
  if (parts.length === 2) return { tab, run };
  if (parts.length !== 3) return null;
  try {
    const file = decodeURIComponent(parts[2]);
    return file ? { tab, run, file } : null;
  } catch {
    return null;
  }
}

export function serialiseSelection(sel: Selection): string {
  if (!sel) return "";
  if (sel.tab === "claims") return "#claims";
  let s = `#${sel.tab}`;
  if (sel.run === undefined) return s;
  s += `/${sel.run}`;
  if (sel.tab === "files" && sel.file) s += `/${encodeURIComponent(sel.file)}`;
  return s;
}

/** Clamp a selection to what the session has: a run that is gone becomes the last run, a file the run lacks is dropped. */
export function resolveSelection(sel: Selection, runs: RunView[]): Selection {
  if (!sel || sel.tab === "claims") return sel;
  if (!runs.length) return { tab: sel.tab };
  const indices = runs.map((r) => r.index);
  const run = sel.run !== undefined && indices.includes(sel.run) ? sel.run : indices[indices.length - 1];
  if (sel.tab === "runs") return { tab: "runs", run };
  const r = runs.find((x) => x.index === run);
  const file = sel.file && r?.files.includes(sel.file) ? sel.file : undefined;
  return file ? { tab: "files", run, file } : { tab: "files", run };
}
