import { useEffect, useState } from "react";
import { parseCsv, type Table } from "../table";
import type { ArtifactView } from "../types";

// An artifact a tool made from the file on request (causal_agent/memory/artifacts.py): the picture when it drew one, the
// summary with the address the chat cites, every number it holds with its own address, and the table when it wrote one,
// its first rows shown here and the whole file one click away.
export default function Picture({ a }: { a: ArtifactView }) {
  const when = a.design == null ? "before any run" : `about design ${a.design}`;
  const [table, setTable] = useState<Table | null>(null);
  useEffect(() => {
    if (!a.table_url) return;
    let live = true;
    fetch(a.table_url)
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => {
        if (live) setTable(parseCsv(text));
      })
      .catch(() => {
        if (live) setTable(null);
      });
    return () => {
      live = false;
    };
  }, [a.table_url]);
  return (
    <figure className="pic" aria-label={a.summary}>
      <figcaption>
        <span className="addr">{a.address}</span> {a.summary} <span className="muted">({when})</span>
      </figcaption>
      {a.url && <img src={a.url} alt={a.summary} loading="lazy" />}
      {Object.keys(a.facts).length > 0 && (
        <dl className="kv facts">
          {Object.entries(a.facts).map(([name, value]) => (
            <div key={name}>
              <dt>
                <span className="addr" title={`${a.address}.${name}`}>
                  {name}
                </span>
              </dt>
              <dd>{Number.isInteger(value) ? value : value.toPrecision(4)}</dd>
            </div>
          ))}
        </dl>
      )}
      {table && table.header.length > 0 && (
        <div className="tbl" aria-label={`${a.address}.table`}>
          <table>
            <thead>
              <tr>
                {table.header.map((h, i) => (
                  <th key={i}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((r, i) => (
                <tr key={i}>
                  {r.map((c, j) => (
                    <td key={j} className="mono">
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {a.table_url && (
        <p className="muted">
          <a href={a.table_url} target="_blank" rel="noreferrer">
            the table: {a.table}
          </a>
          {table && table.total > table.rows.length ? ` (${table.rows.length} of ${table.total} rows shown)` : ""}
        </p>
      )}
    </figure>
  );
}
