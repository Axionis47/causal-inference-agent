import { useState } from "react";
import type { components } from "../generated/schema";
type Wire<K extends keyof components["schemas"]> = components["schemas"][K];

export default function AddContextStatement({
  section,
  columns,
  disabled,
  save,
}: {
  section: string;
  columns: Wire<"ColumnIdentity">[];
  disabled: boolean;
  save: (edits: Wire<"ContextEdit">[]) => Promise<boolean>;
}) {
  const [open, setOpen] = useState(false),
    [value, setValue] = useState(""),
    [field, setField] = useState("");
  const [from, setFrom] = useState(""),
    [to, setTo] = useState(""),
    [timing, setTiming] = useState(""),
    [conditions, setConditions] = useState("");
  const [reason, setReason] = useState("");
  const column = columns.find((c) => c.column_id === section);
  const relationship = section === "relationship";
  const fields = relationship
    ? [
        "association",
        "causal_influence",
        "no_causal_influence",
        "temporal_order",
        "measurement",
        "derivation",
        "assignment_rule",
        "selection_rule",
        "membership",
      ]
    : column
      ? ["definition", "collection_time", "reference_period", "codes", "units", "missing_codes", "construction", "qualifications"]
      : ["observations", "population", "organization", "study_process", "measurement", "qualifications"];
  if (section === "requirement") return null;
  return (
    <section>
      <button disabled={disabled} onClick={() => setOpen(!open)}>
        {open ? "Close new statement" : "Add context statement"}
      </button>
      {open && (
        <fieldset disabled={disabled}>
          <legend>Add supplied knowledge</legend>
          <label>
            Statement type
            <select value={field} onChange={(e) => setField(e.target.value)}>
              <option value="">Choose a type</option>
              {fields.map((f) => (
                <option key={f} value={f}>
                  {f.replaceAll("_", " ")}
                </option>
              ))}
            </select>
          </label>
          <label>
            Statement
            <textarea value={value} onChange={(e) => setValue(e.target.value)} />
          </label>
          {relationship && (
            <>
              <label>
                From column
                <select value={from} onChange={(e) => setFrom(e.target.value)}>
                  <option value="">Choose a column</option>
                  {columns.map((c) => (
                    <option key={c.column_id} value={c.column_id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                To column
                <select value={to} onChange={(e) => setTo(e.target.value)}>
                  <option value="">Choose a column</option>
                  {columns.map((c) => (
                    <option key={c.column_id} value={c.column_id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Timing
                <input value={timing} onChange={(e) => setTiming(e.target.value)} />
              </label>
              <label>
                Conditions
                <input value={conditions} onChange={(e) => setConditions(e.target.value)} />
              </label>
              <p>Only state a causal relationship if you can confirm it. An association alone does not establish causation.</p>
            </>
          )}
          <label>
            Source or reason
            <input value={reason} onChange={(e) => setReason(e.target.value)} />
          </label>
          <button
            disabled={!field || !value.trim() || !reason.trim() || (relationship && (!from || !to || from === to))}
            onClick={async () => {
              const accepted = await save([
                {
                  action: "add",
                  reason,
                  statement: {
                    id: crypto.randomUUID(),
                    basis: "source_statement",
                    scope: relationship ? "relationship" : column ? "column" : "dataset",
                    field,
                    value,
                    column_ids: column ? [column.column_id] : [],
                    participants: relationship
                      ? [
                          { kind: "column", id: from, role: "from" },
                          { kind: "column", id: to, role: "to" },
                        ]
                      : [],
                    timing,
                    conditions,
                    qualifications: "",
                    directness: "unspecified",
                    absence_scope: "unspecified",
                    status: "reviewed",
                    reason,
                    source: { kind: "review", id: "displayed-edit", text: reason },
                  },
                },
              ]);
              if (!accepted) return;
              setOpen(false);
              setValue("");
              setReason("");
            }}
          >
            Save statement
          </button>
        </fieldset>
      )}
    </section>
  );
}
