import type { ColumnSummary } from "../types";

export default function ColumnForm({
  columns,
  values,
  onChange,
}: {
  columns: ColumnSummary[];
  values: Record<string, string>;
  onChange: (name: string, v: string) => void;
}) {
  return (
    <div className="cols">
      {columns.map((c) => (
        <div className="colrow" key={c.name}>
          <div>
            <div className="cname">{c.name}</div>
            <span className="ex" title={c.examples.join(", ")}>
              {c.examples.length ? c.examples.join(" · ") : `${c.distinct} distinct`}
              {c.nulls ? ` · ${c.nulls} missing` : ""}
            </span>
          </div>
          <span className="pill">{c.kind}</span>
          <input
            value={values[c.name] ?? ""}
            placeholder="One line: what it records, and when it was set"
            onChange={(e) => onChange(c.name, e.target.value)}
            aria-label={`Description of ${c.name}`}
          />
        </div>
      ))}
    </div>
  );
}
