import { statusMatrix } from "../../inspector/rows";
import type { ClaimView, StatusView } from "../../types";

const WORD: Record<string, string> = { fits: "fits", does_not_fit: "does not fit", unknown: "unknown", not_needed: "·" };

/** Claim kinds down the side, designs across the top: what each claim kind says about each design. */
export default function StatusMatrix({ status, claims }: { status: StatusView | null; claims: ClaimView[] }) {
  const m = statusMatrix(status, claims);
  if (!m || !status) return null;
  return (
    <section>
      <h3>What each claim kind says about each design</h3>
      <div className="tbl matrix">
        <table>
          <thead>
            <tr>
              <th>kind</th>
              {m.families.map((f) => (
                <th key={f.name} className={f.struck ? "strike" : undefined} title={f.struck ? `struck out: ${f.struck}` : undefined}>
                  {f.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {m.rows.map((r) => (
              <tr key={r.kind}>
                <td className="k">{r.kind.replace(/_/g, " ")}</td>
                {m.families.map((f) => (
                  <td key={f.name} data-cell={r.cells[f.name]}>
                    {WORD[r.cells[f.name]] ?? r.cells[f.name]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {status.contradictions.length > 0 && <p className="err">Contradictions: {status.contradictions.join("; ")}</p>}
    </section>
  );
}
