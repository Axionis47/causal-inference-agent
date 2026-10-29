import { useState } from "react";
import { api } from "../api";
import type { components } from "../generated/schema";
import { parseTable, type Table } from "../table";
import Figure from "./Figure";
import type { FigureSpec } from "../types";

type Wire<K extends keyof components["schemas"]> = components["schemas"][K];

export default function ArtifactCard({
  chat,
  item,
  earlier,
  disabled,
  discuss,
}: {
  chat: string;
  item: Wire<"ArtifactSummary">;
  earlier: boolean;
  disabled: boolean;
  discuss: (id: string, element?: string) => Promise<boolean>;
}) {
  const [detail, setDetail] = useState<Wire<"Artifact"> | null>(null);
  const [table, setTable] = useState<Table | null>(null);
  const [figures, setFigures] = useState<FigureSpec[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  async function inspect() {
    setLoading(true);
    setError("");
    try {
      const a = await api.artifact(chat, item.id);
      setDetail(a);
      if (a.table) {
        const r = await fetch(api.artifactFile(chat, a.id, a.table));
        if (!r.ok) throw new Error("Table unavailable");
        setTable(parseTable(a.table, await r.text()));
      }
      if (a.files.includes("figures.json")) {
        const r = await fetch(api.artifactFile(chat, a.id, "figures.json"));
        if (!r.ok) throw new Error("Figures unavailable");
        setFigures(await r.json());
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }
  return (
    <article className="stage">
      <h2>{item.summary}</h2>
      <p>Request: {item.ask}</p>
      <p>
        {item.origin === "analysis" ? "Causal analysis" : "Exploration"} · {item.format} · {new Date(item.made_at).toLocaleString()}
      </p>
      {earlier && <p>Produced in an earlier chat. Discussing it records reuse here and keeps its original provenance.</p>}
      {item.url && <img src={item.url} alt={item.summary} loading="lazy" style={{ maxWidth: "100%" }} />}
      <button disabled={disabled} onClick={() => void discuss(item.id)}>
        Discuss artifact
      </button>
      {!earlier && !detail && (
        <button disabled={loading} onClick={() => void inspect()}>
          {loading ? "Loading evidence…" : "View evidence and files"}
        </button>
      )}
      {error && <p role="alert">{error}</p>}
      {detail && (
        <>
          <p>{detail.method}</p>
          <ul>
            {detail.findings.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
          <details>
            <summary>Limitations and provenance</summary>
            {detail.limitations.map((x, i) => (
              <p key={i}>{x}</p>
            ))}
            <p>Request: {detail.ask}</p>
            <p>Produced {new Date(detail.made_at).toLocaleString()}. Dataset version and original producer are retained with this artifact.</p>
          </details>
          {Object.entries(detail.facts).length > 0 && (
            <dl>
              {Object.entries(detail.facts).map(([key, value]) => (
                <div key={key}>
                  <dt>{key.replaceAll("_", " ")}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
          )}
          {figures.map((f) => (
            <Figure key={f.id} spec={f} />
          ))}
          {table && (
            <>
              <p>Preview of {table.total} table rows.</p>
              <div className="table-scroll">
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
                        {r.map((v, j) => (
                          <td key={j}>{v}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {detail.elements?.map((e) => (
            <button key={e.id} disabled={disabled} onClick={() => void discuss(detail.id, e.id)}>
              Discuss {e.label}
            </button>
          ))}
          <details>
            <summary>Registered files</summary>
            <ul>
              {detail.files.map((file) => (
                <li key={file}>
                  <a href={api.artifactFile(chat, detail.id, file)}>{file}</a>
                </li>
              ))}
            </ul>
          </details>
        </>
      )}
    </article>
  );
}
