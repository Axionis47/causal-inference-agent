import type { ColumnSummary, ProfileOut } from "../types";

const n = (v: number) => v.toLocaleString();
const num = (v: number) => (Number.isInteger(v) ? n(v) : v.toLocaleString(undefined, { maximumFractionDigits: 2 }));
const pct = (v: number) => `${(v * 100).toLocaleString(undefined, { maximumFractionDigits: v < 0.01 ? 1 : 0 })}%`;

export function grainText(p: ProfileOut): string {
  if (p.grain) return `One row per ${p.grain.join(" + ")}.`;
  if (p.candidate_keys.length) return `One row per ${p.candidate_keys[0].join(" + ")}.`;
  return "No column, or pair of columns, identifies a row.";
}

export function datasetWarnings(p: ProfileOut): string[] {
  const out = [...p.issues];
  if (p.duplicate_rows) out.push(`${n(p.duplicate_rows)} duplicate ${p.duplicate_rows === 1 ? "row" : "rows"}`);
  for (const [a, b] of p.co_missing) out.push(`${a} and ${b} are missing on the same rows`);
  return out;
}

export function columnWarnings(c: ColumnSummary): string[] {
  const out = [...c.issues];
  if (c.constant) out.push("every row has the same value");
  for (const s of c.sentinels) out.push(`${s.value} × ${n(s.count)}: ${s.reason}`);
  return out;
}

function Quartiles({ q }: { q: NonNullable<ColumnSummary["numeric"]> }) {
  const span = q.max - q.min || 1;
  const at = (v: number) => `${((v - q.min) / span) * 100}%`;
  return (
    <div className="shape">
      <div
        className="qbar"
        title={`min ${num(q.min)} · p25 ${num(q.p25)} · median ${num(q.p50)} · p75 ${num(q.p75)} · max ${num(q.max)} · mean ${num(q.mean)}`}
      >
        <span className="box" style={{ left: at(q.p25), width: `calc(${at(q.p75)} - ${at(q.p25)})` }} />
        <span className="mid" style={{ left: at(q.p50) }} />
      </div>
      <span className="ex">
        {num(q.min)} · <b>{num(q.p50)}</b> · {num(q.max)}
      </span>
    </div>
  );
}

function TopValues({ c }: { c: ColumnSummary }) {
  const shown = c.top_values.slice(0, 4);
  const rest = c.distinct - shown.length;
  return (
    <div className="shape bars">
      {shown.map((t) => (
        <div className="bar" key={t.value} title={`${t.value}: ${n(t.count)} rows`}>
          <span className="fill" style={{ width: pct(t.share) }} />
          <span className="lbl">{t.value || "(blank)"}</span>
          <span className="val">{pct(t.share)}</span>
        </div>
      ))}
      {rest > 0 && <span className="ex">and {n(rest)} more</span>}
    </div>
  );
}

function Shape({ c }: { c: ColumnSummary }) {
  if (c.numeric && (c.kind === "numeric" || c.kind === "id")) return <Quartiles q={c.numeric} />;
  if (c.top_values.length) return <TopValues c={c} />;
  if (c.datetime)
    return (
      <div className="shape">
        <span className="ex">
          {c.datetime.first} → {c.datetime.last}
          {c.datetime.frequency ? ` · ${c.datetime.frequency}` : ""}
        </span>
      </div>
    );
  return (
    <div className="shape">
      <span className="ex">{c.examples.length ? c.examples.join(" · ") : `${n(c.distinct)} distinct values`}</span>
    </div>
  );
}

export default function DatasetPreview({ profile }: { profile: ProfileOut }) {
  const warnings = datasetWarnings(profile);
  const names = profile.columns.map((c) => c.name);
  return (
    <div className="preview">
      <div className="facts">
        <span>
          <b>{n(profile.rows)}</b> rows
        </span>
        <span>
          <b>{profile.columns.length}</b> columns
        </span>
        <span>{grainText(profile)}</span>
        {warnings.map((w) => (
          <span className="pill amber" key={w}>
            {w}
          </span>
        ))}
      </div>

      <div className="head" role="region" aria-label="First rows of the file">
        <table>
          <thead>
            <tr>
              {names.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profile.head.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} className={cell === "" ? "blank" : undefined}>
                    {cell === "" ? "·" : cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="ex under">
        The first {profile.head.length} of {n(profile.rows)} rows, as read.
      </p>

      <div className="colshape">
        {profile.columns.map((c) => {
          const warn = columnWarnings(c);
          return (
            <div className="colrow" key={c.name}>
              <div>
                <div className="cname">{c.name}</div>
                <span className="ex">
                  {n(c.distinct)} distinct{c.nulls ? ` · ${n(c.nulls)} missing (${pct(c.null_rate)})` : ""}
                </span>
                {warn.length > 0 && (
                  <div className="warns">
                    {warn.map((w) => (
                      <span className="pill amber" key={w}>
                        {w}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <span className="pill">{c.kind}</span>
              <Shape c={c} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
