// The handoff view's rows: the claim fields in their own group, then one group per column, in the order the pack lists them.
import type { HandoffField } from "./types";

export interface FieldGroup {
  title: string;
  fields: HandoffField[];
}

export function fieldGroups(fields: HandoffField[]): FieldGroup[] {
  const groups: FieldGroup[] = [];
  const byTitle = new Map<string, FieldGroup>();
  for (const f of fields) {
    const title = f.address.startsWith("claim:") ? "claims" : f.address.slice(0, f.address.lastIndexOf("."));
    let g = byTitle.get(title);
    if (!g) {
      g = { title, fields: [] };
      byTitle.set(title, g);
      groups.push(g);
    }
    g.fields.push(f);
  }
  return groups;
}

/** A field's value as text the person can read and edit back: lists joined, booleans as words, nothing as an empty string. */
export function valueText(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (Array.isArray(v)) return v.map(String).join(", ");
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
