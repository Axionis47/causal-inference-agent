import { useState } from "react";
import type { components } from "../generated/schema";

type Statement = components["schemas"]["KnowledgeStatement"];
type Edit = components["schemas"]["ContextEdit"];
type Column = components["schemas"]["ColumnIdentity"];

export default function ContextStatements({
  statements,
  columns,
  onEdit,
  disabled = false,
}: {
  statements: Statement[];
  columns: Column[];
  onEdit: (edits: Edit[]) => Promise<boolean>;
  disabled?: boolean;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState<Statement | null>(null);
  const [reason, setReason] = useState("");
  const labels = new Map(columns.map((c) => [c.column_id, c.name]));
  return (
    <div className="context-statements">
      {!statements.length && <p>No supplied statements in this section. Leave missing information unresolved.</p>}
      {statements.some((s) => s.status !== "reviewed") && (
        <button
          disabled={disabled}
          onClick={() =>
            void onEdit(
              statements
                .filter((s) => s.status !== "reviewed")
                .map((s) => ({ action: "review", target_id: s.id, reason: "Confirmed the displayed statement." })),
            )
          }
        >
          Confirm the displayed statements
        </button>
      )}
      {statements.map((s) => (
        <article key={s.id} className="stage">
          <strong>{s.scope === "requirement" ? "Analysis requirement" : s.field.replaceAll("_", " ")}</strong>
          <span className="pill">{s.status}</span>
          <p>{s.value}</p>
          {s.qualifications && <p>Qualification: {s.qualifications}</p>}
          {(s.participants?.length ?? 0) > 0 && <p>{s.participants!.map((p) => `${labels.get(p.id) ?? "Supplied concept"} (${p.role})`).join(", ")}</p>}
          {s.timing && <p>Timing: {s.timing}</p>}
          {s.conditions && <p>Conditions: {s.conditions}</p>}
          {s.field === "no_causal_influence" && <p>Absence scope: {s.absence_scope?.replaceAll("_", " ")}</p>}
          {s.citations?.map((c, i) => (
            <blockquote key={i}>
              <p>{c.quote}</p>
              <small>Original account, line {c.source_line}</small>
            </blockquote>
          ))}
          <small>
            Source: {s.source.kind === "extraction" ? "Mapped from the original account" : s.source.kind === "review" ? "Context review" : "Chat correction"}
          </small>
          {s.source.text && <p>{s.source.text}</p>}
          <div className="actions">
            <button
              disabled={disabled}
              onClick={() => {
                setEditing(s.id);
                setDraft({ ...s });
                setReason("");
              }}
            >
              Edit statement
            </button>
            <button disabled={disabled} onClick={() => void onEdit([{ action: "unresolved", target_id: s.id, reason: "Left unresolved during review." }])}>
              Leave unresolved
            </button>
          </div>
          {editing === s.id && draft && (
            <fieldset disabled={disabled}>
              <legend>Edit {s.field.replaceAll("_", " ")}</legend>
              <label>
                Description
                <textarea
                  aria-label="Description"
                  value={draft.value}
                  onChange={(e) => setDraft({ ...draft, value: e.target.value, binding: draft.binding ? { ...draft.binding, value: e.target.value } : null })}
                />
              </label>
              <label>
                Qualification
                <textarea value={draft.qualifications ?? ""} onChange={(e) => setDraft({ ...draft, qualifications: e.target.value })} />
              </label>
              {s.scope === "relationship" && (
                <>
                  {draft.participants?.map((p, i) => (
                    <div key={i}>
                      {p.kind === "column" && (
                        <label>
                          Participant {i + 1}
                          <select
                            value={p.id}
                            onChange={(e) =>
                              setDraft({ ...draft, participants: draft.participants!.map((v, j) => (j === i ? { ...v, id: e.target.value } : v)) })
                            }
                          >
                            {columns.map((c) => (
                              <option key={c.column_id} value={c.column_id}>
                                {c.name}
                              </option>
                            ))}
                          </select>
                        </label>
                      )}
                      <label>
                        Relationship role
                        <select
                          value={p.role}
                          onChange={(e) =>
                            setDraft({
                              ...draft,
                              participants: draft.participants!.map((v, j) => (j === i ? { ...v, role: e.target.value as typeof p.role } : v)),
                            })
                          }
                        >
                          {["participant", "from", "to", "intermediate", "condition"].map((role) => (
                            <option key={role}>{role}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                  ))}
                  <label>
                    Timing
                    <input value={draft.timing ?? ""} onChange={(e) => setDraft({ ...draft, timing: e.target.value })} />
                  </label>
                  <label>
                    Conditions
                    <input value={draft.conditions ?? ""} onChange={(e) => setDraft({ ...draft, conditions: e.target.value })} />
                  </label>
                  <label>
                    Directness
                    <select value={draft.directness} onChange={(e) => setDraft({ ...draft, directness: e.target.value as Statement["directness"] })}>
                      <option value="unspecified">Unspecified</option>
                      <option value="direct">Direct</option>
                      <option value="indirect">Indirect</option>
                    </select>
                  </label>
                  {s.field === "no_causal_influence" && (
                    <label>
                      Absence scope
                      <select value={draft.absence_scope} onChange={(e) => setDraft({ ...draft, absence_scope: e.target.value as Statement["absence_scope"] })}>
                        <option value="unspecified">Unspecified</option>
                        <option value="direct_effect">Direct effect</option>
                        <option value="any_pathway">Any pathway</option>
                      </select>
                    </label>
                  )}
                </>
              )}
              <label>
                Reason for the change
                <input value={reason} onChange={(e) => setReason(e.target.value)} />
              </label>
              <button
                disabled={!reason.trim() || !draft.value.trim()}
                onClick={async () => {
                  if (await onEdit([{ action: "replace", target_id: s.id, statement: { ...draft, id: crypto.randomUUID() }, reason }])) setEditing(null);
                }}
              >
                Save correction
              </button>
              <button
                disabled={!reason.trim()}
                onClick={async () => {
                  if (await onEdit([{ action: "retire", target_id: s.id, reason }])) setEditing(null);
                }}
              >
                Withdraw statement
              </button>
              <button onClick={() => setEditing(null)}>Cancel</button>
            </fieldset>
          )}
        </article>
      ))}
    </div>
  );
}
