import type { ArtifactView } from "../types";

// An artifact a tool made from the file on request (causal_agent/memory/artifacts.py): the picture when it drew one, the
// summary with the address the chat cites, every number it holds with its own address, and the table when it wrote one.
export default function Picture({ a }: { a: ArtifactView }) {
  const when = a.design == null ? "before any run" : `about design ${a.design}`;
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
      {a.table_url && (
        <p className="muted">
          <a href={a.table_url} target="_blank" rel="noreferrer">
            the table: {a.table}
          </a>
        </p>
      )}
    </figure>
  );
}
