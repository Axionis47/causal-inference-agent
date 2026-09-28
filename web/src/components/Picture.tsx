import type { ArtifactView } from "../types";

// A picture the drawing tool made on request (causal_agent/viz/store.py): the image, its caption with the address the chat
// cites, and every number it holds with its own address.
export default function Picture({ a }: { a: ArtifactView }) {
  const when = a.moment === "pre" ? "before the run" : `after design ${a.design}`;
  return (
    <figure className="pic" aria-label={a.caption}>
      <figcaption>
        <span className="addr">{a.address}</span> {a.caption} <span className="muted">({when})</span>
      </figcaption>
      <img src={a.url} alt={a.caption} loading="lazy" />
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
    </figure>
  );
}
