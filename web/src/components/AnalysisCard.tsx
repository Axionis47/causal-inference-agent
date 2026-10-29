import { useState } from "react";
import { api } from "../api";
import type { components } from "../generated/schema";
import type { RunView } from "../types";
import Figure from "./Figure";

export default function AnalysisCard({ chat, item }: { chat: string; item: components["schemas"]["AnalysisSummary"] }) {
  const [detail, setDetail] = useState<RunView | null>(null);
  const [error, setError] = useState("");
  return (
    <article className="stage">
      <h2>
        Analysis {item.index}: {item.status.replaceAll("_", " ")}
      </h2>
      <p>{item.question}</p>
      <p>{item.family?.replaceAll("_", " ")}</p>
      {!detail && (
        <button
          onClick={() =>
            void api
              .analysis(chat, item.id)
              .then(setDetail)
              .catch((e) => setError(String(e)))
          }
        >
          View analysis details
        </button>
      )}
      {error && <p role="alert">{error}</p>}
      {detail && (
        <>
          <p>
            {detail.effect === null ? "No lead estimate is available. A successful companion does not replace a failed lead." : `Lead effect: ${detail.effect}`}
          </p>
          {detail.context_deficiency && (
            <p role="alert">{detail.context_deficiency.reason} This attempt stopped. Correct the context, then prepare and authorize a new snapshot.</p>
          )}
          {detail.feasibility && <p>{detail.feasibility.reason}</p>}
          {detail.estimates.map((e, i) => (
            <section key={i}>
              <strong>{e.method}</strong>
              <p>{e.error ?? `Estimate: ${e.value}`}</p>
              {e.confidence_set && <p>{e.confidence_set.detail}</p>}
              {e.ci_low !== null && e.ci_high !== null && (
                <p>
                  Interval: {e.ci_low} to {e.ci_high}
                </p>
              )}
            </section>
          ))}
          {detail.interpretations.map((x, i) => (
            <section key={i}>
              <p>{x.answer}</p>
              {x.caveats.map((c, j) => (
                <p key={j}>{c}</p>
              ))}
            </section>
          ))}
          <details>
            <summary>Checks and limitations</summary>
            {detail.checks.map((c, i) => (
              <p key={i}>
                {c.name}: {c.level}, {c.detail}
              </p>
            ))}
            {detail.declines.map((d, i) => (
              <p key={i}>{d.reason}</p>
            ))}
          </details>
          {detail.figures.map((f) => (
            <Figure key={f.id} spec={f} />
          ))}
        </>
      )}
    </article>
  );
}
