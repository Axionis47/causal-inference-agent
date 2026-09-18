import { Fragment } from "react";
import { ci, count, num } from "../../fmt";
import { checkRows, decisionOver, decisionWhy, estimateRows, primaryEstimate, refutationRows } from "../../inspector/rows";
import type { Selection } from "../../selection";
import type { RunView } from "../../types";
import { ChecksTable, EstimatesTable, RefutationsTable } from "./RunTables";

export default function RunDetail({ r, prev, onSelect }: { r: RunView; prev: RunView | null; onSelect: (s: Selection) => void }) {
  const primary = primaryEstimate(r);
  const why = decisionWhy(r);
  const over = decisionOver(r);
  return (
    <>
      <section className="run-sum">
        <div className="h">
          <h3>
            Run {r.index} · {r.family ?? "no design"}
            {r.specialist && <span className="muted"> via {r.specialist}</span>}
          </h3>
          <span className="pill">{r.status}</span>
        </div>
        <p className="dim">{r.question}</p>
        {r.effect !== null ? (
          <div className="eff">
            effect <span className="v">{num(r.effect)}</span> {ci(r.ci_low, r.ci_high)}
            {r.estimator ? ` by ${r.estimator}` : ""}
            {primary && primary.n_treated != null && (
              <span className="muted">
                {" "}
                · {count(primary.n_treated)} treated, {count(primary.n_control)} control
              </span>
            )}
          </div>
        ) : (
          <div className="eff muted">no estimate</div>
        )}
        {prev && prev.effect !== null && r.effect !== null && (
          <div className="eff muted">
            then {num(prev.effect)} · now {num(r.effect)}
          </div>
        )}
        {r.feasibility && (
          <p className="dim">
            stopped at {String(r.feasibility.stage)}: {String(r.feasibility.reason)}
          </p>
        )}
      </section>

      <EstimatesTable rows={estimateRows(r)} />
      <ChecksTable rows={checkRows(r.flags)} title="Flags" />
      <ChecksTable rows={checkRows(r.checks)} title="Checks" />
      <RefutationsTable rows={refutationRows(r)} />

      <section>
        <h3>Why this design</h3>
        {why ? <p>{why}</p> : <p className="muted">no reason recorded</p>}
        {over.length > 0 && (
          <dl className="kv">
            {over.map((o) => (
              <Fragment key={o.family}>
                <dt>{o.family}</dt>
                <dd>{o.reason}</dd>
              </Fragment>
            ))}
          </dl>
        )}
      </section>

      {r.interpretations.length > 0 && (
        <section>
          <h3>The lane's reading</h3>
          {r.interpretations.map((it, i) => (
            <div key={i} className="reading">
              {it.contrast && <div className="mono muted">{it.contrast}</div>}
              <p>{it.answer}</p>
              {it.caveats.length > 0 && (
                <ul>
                  {it.caveats.map((c, j) => (
                    <li key={j}>{c}</li>
                  ))}
                </ul>
              )}
              {it.cites.length > 0 && <div className="cites">{it.cites.join(" · ")}</div>}
            </div>
          ))}
        </section>
      )}

      {r.status === "pipeline_error" && (
        <section>
          <details>
            <summary>What the process said</summary>
            <pre className="mono pre">{r.decision_record}</pre>
          </details>
        </section>
      )}

      {r.files.length > 0 && (
        <section>
          <h3>Files</h3>
          <div className="chips">
            {r.files.map((f) => (
              <button key={f} type="button" className="chip" onClick={() => onSelect({ tab: "files", run: r.index, file: f })}>
                {f}
              </button>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
