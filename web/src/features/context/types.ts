import type { components } from "@/lib/api/schema";
export type Workspace = components["schemas"]["WorkspaceView"];
export type Content = components["schemas"]["ContextContent"];
export type Claim = components["schemas"]["Statement"];
export type DatasetClaim = components["schemas"]["DatasetStatement"];
export type Concept = components["schemas"]["Concept"];
export type Relation = components["schemas"]["Relationship"];
export type Citation = components["schemas"]["Citation"];
export type Edit = components["schemas"]["ContextEdit"];
export type EditStatement = components["schemas"]["EditStatement"];
export type Source = components["schemas"]["SourceView"];
export type Read = components["schemas"]["RetrievedContext"];
export type Entry = {
  key: string;
  claim: Claim | DatasetClaim | Concept | Relation;
};
export const label = (key: string) =>
  key.replaceAll("_", " ").replace(/^./, (s) => s.toUpperCase());
export function entries(
  content: Content,
  section: string,
  workspace: Workspace,
): Entry[] {
  if (section === "concepts" || section === "relationships") {
    return Object.entries(content[section] ?? {}).map(([id, claim]) => ({
      key: `${section}.${id}`,
      claim,
    }));
  }
  const column = section.startsWith("columns.")
    ? content.columns?.[section.split(".")[1] ?? ""]
    : undefined;
  const target = column ?? content.dataset;
  if (!target) return [];
  const fields = column ? workspace.column_fields : workspace.dataset_fields;
  return fields.flatMap((field) => {
    const records = target[field as keyof typeof target] as
      Record<string, Claim> | undefined;
    return Object.entries(records ?? {}).map(([id, claim]) => ({
      key: `${section}.${field}.${id}`,
      claim,
    }));
  });
}

export function entrySection(key: string): string {
  const parts = key.split(".");
  return parts[0] === "columns" ? parts.slice(0, 2).join(".") : parts[0]!;
}
export const columnGroups = [
  { title: "Meaning", fields: ["definition", "concept", "entity"] },
  {
    title: "Values",
    fields: ["units", "codes", "valid_ranges", "missing_codes"],
  },
  {
    title: "Measurement",
    fields: [
      "method",
      "measurement_source",
      "determined_by",
      "collection_time",
      "reference_period",
    ],
  },
  { title: "Construction", fields: ["construction", "derivation"] },
  { title: "Missingness", fields: ["missingness"] },
  { title: "Qualifications", fields: ["qualifications"] },
] as const;
