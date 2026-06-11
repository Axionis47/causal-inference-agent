import { Link } from "@tanstack/react-router";
import type { components } from "@/lib/api/schema";

export function AnalysisResults({
  data,
  datasetId,
  versionId,
  conversationId,
}: {
  data: components["schemas"]["AnalysisView"];
  datasetId: string;
  versionId: string;
  conversationId: string;
}) {
  const result = data.result;
  if (!result) return null;
  return (
    <section className="analysis-section" aria-label="Saved analysis results">
      <h3 className="eyebrow">
        {result.execution_ok ? "Saved results" : "Saved failure report"}
      </h3>
      <p className="whitespace-pre-wrap mt-3">{result.report.text}</p>
      {!!data.snapshot?.plan.limitations.length && (
        <div className="subtext mt-3">
          <p className="eyebrow">Interpretation limits</p>
          {data.snapshot.plan.limitations.map((limitation) => (
            <p className="mt-2" key={limitation}>
              {limitation}
            </p>
          ))}
        </div>
      )}
      {result.filenames.includes("figure.png") && (
        <img
          className="w-full max-h-[440px] object-contain mt-4"
          src={`/api/analyses/${data.id}/files/figure.png`}
          alt="Estimate and uncertainty from the approved analysis"
        />
      )}
      <div className="flex flex-wrap gap-3 mt-4">
        {result.filenames.map((name) => (
          <FileLink key={name} run={data.id} name={name}>
            {name}
          </FileLink>
        ))}
      </div>
      <details className="mt-4">
        <summary>Exact computed values</summary>
        <dl className="analysis-facts">
          {Object.entries(result.facts).map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>
                {value.toLocaleString(undefined, {
                  maximumSignificantDigits: 7,
                })}
              </dd>
            </div>
          ))}
        </dl>
      </details>
      <Link
        className="context-link inline-block mt-4"
        to="/datasets/$datasetId/chat/$versionId"
        params={{ datasetId, versionId }}
        search={{
          conversation: conversationId,
          analysis: data.id,
          view: "chat",
        }}
      >
        Discuss these results in chat
      </Link>
    </section>
  );
}

export function FileLink({
  run,
  name,
  children,
}: {
  run: string;
  name: string;
  children: React.ReactNode;
}) {
  return (
    <a
      className="context-link"
      href={`/api/analyses/${run}/files/${encodeURIComponent(name)}`}
      download
    >
      {children}
    </a>
  );
}
