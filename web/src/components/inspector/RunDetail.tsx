import { Fragment } from "react";
import { ci, count, num } from "../../fmt";
import { checkRows, decisionOver, decisionWhy, declineRows, estimateRows, primaryEstimate, falsificationRows } from "../../inspector/rows";
import type { Selection } from "../../selection";
import type { FigureSpec } from "../../figure";
import type { RunView } from "../../types";
import Figure from "../Figure";
import Picture from "../Picture";
import { ChecksTable, DeclinesTable, EstimatesTable, FalsificationsTable } from "./RunTables";

// The figures the run drew from its own artifacts, in the order the chat shows them.
export function Figures({ figures }: { figures: FigureSpec[] }) {
  if (!figures.length) return null;
  return (
    <section>
      <h3>Figures</h3>
      {figures.map((f) => (
        <Figure key={f.id} spec={f} />
      ))}
    </section>
  );
}

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
        {Object.keys(r.what_if ?? {}).length > 0 && (
          <p className="dim">
            A what-if: nothing known changed. The copy differed in{" "}
            {Object.entries(r.what_if).map(([a, v]) => (
              <span key={a}>
                <span className="addr">{a}</span> = {v}{" "}
              </span>
            ))}
          </p>
        )}
        {(r.differs ?? []).length > 0 && (
          <p className="dim">
            Differs from the design before in{" "}
            {r.differs.map((a) => (
              <span key={a} className="addr">
                {a}{" "}
              </span>
            ))}
          </p>
        )}
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
        {prev && prev.context_id === r.context_id && prev.effect !== null && r.effect !== null && (
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

      <section>
        <h3>Admitted context and designs</h3>
        <p className="mono">
          Context {r.context_id || "unavailable"}. Lead {r.lead_design_id || "none"}.
        </p>
        {r.context_deficiency && (
          <p className="err">
            {r.context_deficiency.reason}: {r.context_deficiency.addresses.join(", ")}
          </p>
        )}
        <details>
          <summary>Context requirements and sources</summary>
          <pre className="mono pre">
            {JSON.stringify({ dataset_identity: r.dataset_identity, admission: r.context_admission, fields: r.context_facts }, null, 2)}
          </pre>
        </details>
        {Object.entries(r.designs).map(([id, design]) => (
          <details key={id}>
            <summary>
              {id === r.lead_design_id ? "Lead: " : "Rival: "}
              {String(design.endpoint_id ?? id)}
            </summary>
            <pre className="mono pre">{JSON.stringify(design, null, 2)}</pre>
          </details>
        ))}
        <details>
          <summary>Candidate decisions and reasoning history</summary>
          <pre className="mono pre">
            {JSON.stringify(
              {
                candidates: r.candidates,
                lead_selection: r.lead_selection,
                candidate_history: r.candidate_history,
                reasoning_history: r.reasoning_history,
                terminal: r.terminal,
                ledger: r.ledger,
                visits: r.episode_visits,
              },
              null,
              2,
            )}
          </pre>
        </details>
        <details>
          <summary>Execution attempts</summary>
          <pre className="mono pre">{JSON.stringify(r.execution_attempts, null, 2)}</pre>
        </details>
      </section>
      {r.comparisons.length > 0 && (
        <section>
          <h3>Design comparisons</h3>
          <pre className="mono pre">{JSON.stringify(r.comparisons, null, 2)}</pre>
        </section>
      )}
      {r.sensitivities.length > 0 && (
        <section>
          <h3>Sensitivities</h3>
          {r.sensitivities.map((s, i) => (
            <details key={i}>
              <summary>
                {s.design_id}: {s.method}
              </summary>
              <p>{s.detail || s.error}</p>
              <pre className="mono pre">{JSON.stringify({ target: s.target, range: [s.range_low, s.range_high], points: s.points }, null, 2)}</pre>
            </details>
          ))}
        </section>
      )}
      <Figures figures={r.figures ?? []} />
      {r.artifacts.length > 0 && (
        <section>
          <h3>Pictures drawn after this run</h3>
          {r.artifacts.map((a) => (
            <Picture key={a.id} a={a} />
          ))}
        </section>
      )}

      <EstimatesTable rows={estimateRows(r)} />
      <ChecksTable rows={checkRows(r.flags)} title="Flags" />
      <DeclinesTable rows={declineRows(r)} />
      <ChecksTable rows={checkRows(r.checks)} title="Checks" />
      <FalsificationsTable rows={falsificationRows(r)} />

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
