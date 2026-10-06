import { useEffect, useState } from "react";
import { fieldGroups, valueText } from "../handoff";
import type { HandoffField, HandoffView } from "../types";

// The context the run would rest on, shown at the ready moment (docs/design.md, "The handoff view"): every field with its
// status, its source and the sentence it rests on. Changing a value sends `address = value` as a message; the desk writes it
// as the person's word and shows the ready moment again. The page holds no value the memory does not.
function Row({ f, disabled, onEdit }: { f: HandoffField; disabled: boolean; onEdit: (address: string, value: string) => void }) {
  const shown = valueText(f.value);
  const [draft, setDraft] = useState(shown);
  useEffect(() => setDraft(shown), [shown]);
  const dirty = draft.trim() !== shown && draft.trim().length > 0;
  const commit = () => {
    if (dirty && !disabled) onEdit(f.address, draft.trim());
  };
  return (
    <tr>
      <td>
        <span className="addr" title={f.reason ?? f.address}>
          {f.address.slice(f.address.lastIndexOf(".") + 1)}
        </span>
      </td>
      <td>
        <input
          className="hv"
          value={draft}
          disabled={disabled}
          aria-label={f.address}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
          }}
        />
        {dirty && (
          <button type="button" className="chip" onClick={commit}>
            change
          </button>
        )}
      </td>
      <td className="mono">{f.status}</td>
      <td className="mono">{f.source ?? "—"}</td>
      <td className="wrap muted">{f.said ?? ""}</td>
    </tr>
  );
}

export default function HandoffPanel({ h, disabled, onEdit }: { h: HandoffView; disabled: boolean; onEdit: (address: string, value: string) => void }) {
  return (
    <section className="handoff" aria-label="What the analysis will rest on">
      <p className="hh">
        <span className="mono">{h.family}</span> via <span className="mono">{h.specialist}</span> · outcome <span className="addr">{h.outcome}</span>
        {h.treatment && (
          <>
            {" "}
            · treatment <span className="addr">{h.treatment}</span>
          </>
        )}
        {h.bets_on && <> · bets on: {h.bets_on}</>}
      </p>
      {fieldGroups(h.fields).map((g) => (
        <div className="tbl" key={g.title}>
          <table>
            <thead>
              <tr>
                <th>{g.title === "claims" ? "claim" : g.title}</th>
                <th>value</th>
                <th>status</th>
                <th>source</th>
                <th>said</th>
              </tr>
            </thead>
            <tbody>
              {g.fields.map((f) => (
                <Row key={f.address} f={f} disabled={disabled} onEdit={onEdit} />
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </section>
  );
}
