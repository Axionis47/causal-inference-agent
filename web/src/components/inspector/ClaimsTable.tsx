import { Fragment } from "react";
import { claimRows } from "../../inspector/rows";
import type { ClaimView } from "../../types";

export default function ClaimsTable({ claims }: { claims: ClaimView[] }) {
  const rows = claimRows(claims);
  if (!rows.length) return <p className="empty-note">No claims yet. They appear once the description has been read.</p>;
  return (
    <section>
      <h3>Every claim, and what it says</h3>
      <div className="tbl">
        <table>
          <thead>
            <tr>
              <th>claim</th>
              <th>kind</th>
              <th>status</th>
              <th>what it says</th>
              <th>source</th>
              <th>evidence</th>
              <th>asked</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <td className="k">{r.key}</td>
                <td className="mono">{r.kind.replace(/_/g, " ")}</td>
                <td>
                  <span className={`status-dot ${r.status}`} />
                  <span className="mono">{r.status}</span>
                  {r.check && <div className="chk">{r.check}</div>}
                </td>
                <td className="wrap">
                  {r.fields.length ? (
                    <dl className="kv">
                      {r.fields.map((f) => (
                        <Fragment key={f.k}>
                          <dt>{f.k}</dt>
                          <dd>{f.v}</dd>
                        </Fragment>
                      ))}
                    </dl>
                  ) : (
                    <span className="muted">no values yet</span>
                  )}
                </td>
                <td className="mono wrap">{r.source ?? "—"}</td>
                <td className="wrap">
                  {r.evidence.length ? (
                    <ul className="ev">
                      {r.evidence.map((e, i) => (
                        <li key={i}>{e}</li>
                      ))}
                    </ul>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="num">
                  {r.asked}
                  {r.refutations ? ` · ${r.refutations} refuted` : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
