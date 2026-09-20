import { count } from "../../fmt";
import type { CheckRow, DeclineRow, EstimateRow, RefutationRow } from "../../inspector/rows";

export function EstimatesTable({ rows }: { rows: EstimateRow[] }) {
  if (!rows.length) return null;
  return (
    <section>
      <h3>Estimates</h3>
      <div className="tbl">
        <table>
          <thead>
            <tr>
              <th>method</th>
              <th>estimate</th>
              <th>95% interval</th>
              <th>n</th>
              <th>treated</th>
              <th>control</th>
              <th>contrast</th>
              <th>note</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={r.primary ? "primary" : r.error ? "err" : undefined}>
                <td className="k">{r.method}</td>
                <td className="num">{r.value}</td>
                <td className="num">{r.ci}</td>
                <td className="num">{count(r.n)}</td>
                <td className="num">{count(r.n_treated)}</td>
                <td className="num">{count(r.n_control)}</td>
                <td className="mono">{r.contrast}</td>
                <td className="wrap">{r.error ?? (r.primary ? "primary" : r.secondary ? "secondary" : "")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function ChecksTable({ rows, title }: { rows: CheckRow[]; title: string }) {
  if (!rows.length) return null;
  return (
    <section>
      <h3>{title}</h3>
      <div className="tbl">
        <table>
          <thead>
            <tr>
              <th>check</th>
              <th>level</th>
              <th>value</th>
              <th>threshold</th>
              <th>contrast</th>
              <th>detail</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="k">{r.name}</td>
                <td className="mono">{r.level}</td>
                <td className="num">{r.value}</td>
                <td className="num">{r.threshold}</td>
                <td className="mono">{r.contrast}</td>
                <td className="wrap">{r.detail || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function RefutationsTable({ rows }: { rows: RefutationRow[] }) {
  const failed = rows.filter((r) => r.passed === "failed").length;
  return (
    <section>
      <h3>Falsifications{rows.length ? ` · ${rows.length - failed} passed${failed ? `, ${failed} failed` : ""}` : ""}</h3>
      {!rows.length ? (
        <p className="muted">none run</p>
      ) : (
        <div className="tbl">
          <table>
            <thead>
              <tr>
                <th>refuter</th>
                <th>kind</th>
                <th>result</th>
                <th>p value</th>
                <th>new effect</th>
                <th>contrast</th>
                <th>detail</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className={r.passed === "failed" ? "fail" : undefined}>
                  <td className="k">{r.refuter}</td>
                  <td className="mono">{r.kind}</td>
                  <td className="mono">{r.passed}</td>
                  <td className="num">{r.p_value}</td>
                  <td className="num">{r.new_effect}</td>
                  <td className="mono">{r.contrast}</td>
                  <td className="wrap">{r.detail || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function DeclinesTable({ rows }: { rows: DeclineRow[] }) {
  if (!rows.length) return null;
  return (
    <section>
      <h3>Where the lane disagreed with the pack</h3>
      <div className="tbl">
        <table>
          <thead>
            <tr>
              <th>about</th>
              <th>kind</th>
              <th>pack said</th>
              <th>lane took</th>
              <th>why</th>
              <th>rule</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.address}>
                <td className="k">
                  <span className="addr">{r.about}</span>
                </td>
                <td className="mono">{r.kind}</td>
                <td className="wrap">{r.packValue}</td>
                <td className="wrap">{r.took}</td>
                <td className="wrap">{r.reason}</td>
                <td className="mono">{r.check}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
