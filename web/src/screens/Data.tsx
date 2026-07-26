/** Screen 2: what the engine read, and what it made of the question.
 *
 *  This screen exists so a wrong reading is caught before any maths happens.
 *  The intake row is the one to check: it says what the engine thinks is doing
 *  the affecting, which is not always a column.
 */
import type { Job } from "../api";
import { Dot, Label, Pane, Row } from "../ui";

export function Data({ job }: { job: Job }) {
  const cols = job.columns ?? [];
  const named = new Set(
    [
      job.intake?.outcome,
      job.intake?.treatment,
      job.intake?.group,
      job.intake?.period,
      job.intake?.time_column,
      job.intake?.running_variable,
    ].filter(Boolean) as string[]
  );

  return (
    <div className="grid grid-cols-[1fr_360px] gap-4 h-full min-h-0">
      <Pane
        caption="columns"
        className="min-h-0 flex flex-col"
        right={
          <span className="text-2xs font-mono text-ink-tertiary tabular">
            {cols.length} cols · {job.n_rows?.toLocaleString()} rows
          </span>
        }
      >
        <div className="overflow-auto max-h-[calc(100vh-220px)]">
          <table className="w-full text-xs font-mono tabular">
            <thead className="sticky top-0 bg-canvas-raised">
              <tr className="text-2xs uppercase tracking-label text-ink-tertiary">
                <th className="text-left font-normal py-1.5 pr-3">name</th>
                <th className="text-left font-normal pr-3">type</th>
                <th className="text-right font-normal pr-3">distinct</th>
                <th className="text-right font-normal pr-3">missing</th>
                <th className="text-right font-normal">range</th>
              </tr>
            </thead>
            <tbody>
              {cols.map((c) => (
                <tr
                  key={c.name}
                  className={`border-t border-edge-subtle/60 ${
                    named.has(c.name) ? "bg-canvas-overlay" : ""
                  }`}
                >
                  <td className="py-1.5 pr-3 flex items-center gap-2">
                    <Dot status={named.has(c.name) ? "info" : "pending"} />
                    <span className={named.has(c.name) ? "text-ink" : "text-ink-secondary"}>
                      {c.name}
                    </span>
                  </td>
                  <td className="pr-3 text-ink-tertiary">
                    {c.datelike ? "date" : c.binary ? "binary" : c.numeric ? "number" : "text"}
                  </td>
                  <td className="text-right pr-3 text-ink-secondary">{c.n_unique}</td>
                  <td
                    className={`text-right pr-3 ${
                      c.missing > 0.3 ? "text-amber" : "text-ink-tertiary"
                    }`}
                  >
                    {c.missing ? `${(c.missing * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="text-right text-ink-tertiary">
                    {c.low !== null ? `${fmt(c.low)} … ${fmt(c.high!)}` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Pane>

      <Pane caption="reading">
        {job.intake ? (
          <div className="space-y-3">
            <div className="space-y-1">
              <Row k="affected by" v={job.intake.exposure} tone="text-amber" />
              <Row k="outcome" v={job.intake.outcome} tone="text-amber" />
              <Row k="question type" v={job.intake.question_family} />
              <Row
                k="confidence"
                v={job.intake.confidence}
                tone={job.intake.confidence === "high" ? "text-mint" : "text-amber"}
              />
            </div>
            <div className="pt-2 space-y-1.5">
              <Label>why</Label>
              <p className="text-xs text-ink-secondary leading-relaxed">
                {job.intake.reasoning}
              </p>
            </div>
            <p className="text-xs text-ink-tertiary leading-relaxed pt-2 border-t border-edge-subtle">
              Not every design has a treatment column. A difference in differences is a
              group crossed with a period; an interrupted series is a date.
            </p>
          </div>
        ) : (
          <p className="text-xs text-ink-tertiary">reading…</p>
        )}
      </Pane>
    </div>
  );
}

function fmt(n: number): string {
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return String(Math.round(n * 100) / 100);
}
