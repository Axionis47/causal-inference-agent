import { journalGroups } from "../../inspector/rows";
import type { Selection } from "../../selection";
import type { StepView } from "../../types";

function when(at: string): string {
  try {
    return new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

/** The conversation's record, one group per design run: the steps that led to it, the design, the run, and what was said about it. */
export default function JournalView({ steps, onSelect }: { steps: StepView[]; onSelect: (s: Selection) => void }) {
  const groups = journalGroups(steps);
  if (!groups.length) return <p className="empty-note">Nothing yet. The journal fills as the conversation reads the question, settles claims, and runs.</p>;
  return (
    <>
      {groups.map((g) => (
        <section key={g.design ?? "pending"}>
          <h3>{g.design === null ? "Leading to the next design" : `Design ${g.design}`}</h3>
          <div className="tbl">
            <table>
              <thead>
                <tr>
                  <th>step</th>
                  <th>what</th>
                  <th>by</th>
                  <th>note</th>
                  <th>when</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {g.rows.map((r) => (
                  <tr key={r.n}>
                    <td className="mono">{r.address}</td>
                    <td className="mono">{r.kind.replace(/_/g, " ")}</td>
                    <td className="mono">{r.by}</td>
                    <td className="wrap">{r.note || "—"}</td>
                    <td className="mono">{when(r.at)}</td>
                    <td>
                      {r.run !== null && (
                        <button type="button" className="chip" onClick={() => onSelect({ tab: "runs", run: r.run ?? undefined })}>
                          Run {r.run}
                        </button>
                      )}{" "}
                      {r.hasFiles && r.run !== null && (
                        <button type="button" className="chip" onClick={() => onSelect({ tab: "files", run: r.run ?? undefined })}>
                          files
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </>
  );
}
