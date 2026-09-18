function Node({ k, v, depth }: { k: string | null; v: unknown; depth: number }) {
  const key = k !== null ? <span className="key">{k}: </span> : null;
  if (v === null || v === undefined) {
    return (
      <div className="kv">
        {key}
        <span className="nil">null</span>
      </div>
    );
  }
  if (typeof v === "number") {
    return (
      <div className="kv">
        {key}
        <span className="num">{Number.isInteger(v) ? v : Number(v).toPrecision(6).replace(/\.?0+$/, "")}</span>
      </div>
    );
  }
  if (typeof v === "boolean") {
    return (
      <div className="kv">
        {key}
        <span className="num">{String(v)}</span>
      </div>
    );
  }
  if (typeof v === "string") {
    return (
      <div className="kv">
        {key}
        <span className="str">"{v}"</span>
      </div>
    );
  }
  const entries = Array.isArray(v) ? v.map((x, i) => [String(i), x] as const) : Object.entries(v as Record<string, unknown>);
  const label = Array.isArray(v) ? `[${entries.length}]` : `{${entries.length}}`;
  return (
    <details open={depth < 1}>
      <summary>
        {key}
        {label}
      </summary>
      {entries.map(([ck, cv]) => (
        <Node key={ck} k={ck} v={cv} depth={depth + 1} />
      ))}
    </details>
  );
}

export default function JsonView({ value }: { value: unknown }) {
  return (
    <div className="json">
      <Node k={null} v={value} depth={0} />
    </div>
  );
}
