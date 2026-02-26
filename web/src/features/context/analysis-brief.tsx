import { Link } from "@tanstack/react-router";
import type { components } from "@/lib/api/schema";

type Brief = components["schemas"]["BriefDraft"];
type Content = components["schemas"]["ContextContent"];
export type ContextLocation = {
  datasetId: string;
  versionId: string;
  revision: string | null;
};

export function ContextReferences({
  references,
  location,
  proposalId,
  content,
  onNavigate,
}: {
  references: string[];
  location: ContextLocation;
  proposalId?: string;
  content?: Content;
  onNavigate?: () => void;
}) {
  return (
    <div className="evidence-links">
      {references.length ? (
        references.map((reference) => {
          const proposed = /^edit:(\d+)$/.exec(reference);
          if (proposed) {
            return proposalId ? (
              <a
                key={reference}
                className="context-link"
                href={`#context-change-${proposalId}-${proposed[1]}`}
              >
                Proposed context change {Number(proposed[1]) + 1}
              </a>
            ) : (
              <span key={reference} className="subtext">
                Unapplied reference: {reference}
              </span>
            );
          }
          const parts = reference.split(".");
          let value: unknown = content;
          for (const part of parts) {
            value =
              value && typeof value === "object"
                ? (value as Record<string, unknown>)[part]
                : undefined;
          }
          const statement =
            value && typeof value === "object" && "value" in value
              ? (value as Record<string, unknown>)
              : undefined;
          const section =
            parts[0] === "columns" && parts[1]
              ? parts.slice(0, 2).join(".")
              : ["concepts", "relationships"].includes(parts[0] ?? "")
                ? parts[0]!
                : "dataset";
          return (
            <div key={reference} className="min-w-0">
              {typeof statement?.value === "string" && (
                <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">
                  {statement.value}
                </p>
              )}
              {typeof statement?.qualifications === "string" &&
                statement.qualifications && (
                  <p className="subtext [overflow-wrap:anywhere]">
                    Qualifications: {statement.qualifications}
                  </p>
                )}
              {typeof statement?.basis === "string" && (
                <p className="subtext">
                  {statement.basis.replaceAll("_", " ")}
                  {typeof statement.status === "string"
                    ? ` / ${statement.status}`
                    : ""}
                </p>
              )}
              <Link
                className="context-link [overflow-wrap:anywhere]"
                title={reference}
                to="/datasets/$datasetId/context/$versionId"
                params={{
                  datasetId: location.datasetId,
                  versionId: location.versionId,
                }}
                search={{ revision: location.revision ?? undefined, section }}
                onClick={onNavigate}
              >
                {reference}
              </Link>
            </div>
          );
        })
      ) : (
        <span className="subtext">No supporting statement selected</span>
      )}
    </div>
  );
}

export function BriefDetails({
  brief,
  location,
  proposalId,
  content,
}: {
  brief: Brief;
  location: ContextLocation;
  proposalId?: string;
  content?: Content;
}) {
  const treatments = brief.treatments ?? [];
  const assumptions = brief.assumptions ?? [];
  const optionalUnknowns = brief.optional_unknowns ?? [];
  const references = (sourceKeys: string[] = []) => (
    <ContextReferences
      references={sourceKeys}
      location={location}
      proposalId={proposalId}
      content={content}
    />
  );
  return (
    <div className="min-w-0" aria-label="Analysis brief details">
      <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">
        {brief.question || "No question recorded"}
      </p>
      <p className="subtext mt-2">
        Supporting statements retain their recorded source and authority.
      </p>
      <dl className="analysis-list">
        <div>
          <dt>Target</dt>
          <dd>
            {brief.target ? (
              <>
                <p className="analysis-hash">
                  Columns:{" "}
                  {(brief.target.column_ids ?? [])
                    .map((id) => content?.columns?.[id]?.identity.name ?? id)
                    .join(", ") || "Not selected"}
                </p>
                {references(brief.target.meaning?.source_keys)}
              </>
            ) : (
              <span className="subtext">Not selected</span>
            )}
          </dd>
        </div>
        <div>
          <dt>Treatments</dt>
          <dd>
            {treatments.length ? (
              <ul className="analysis-list">
                {treatments.map((treatment) => (
                  <li key={treatment.id}>
                    <strong>{treatment.id}</strong>
                    <p className="analysis-hash">
                      Columns:{" "}
                      {(treatment.column_ids ?? [])
                        .map(
                          (id) => content?.columns?.[id]?.identity.name ?? id,
                        )
                        .join(", ") || "Not selected"}
                    </p>
                    <p className="subtext">Meaning</p>
                    {references(treatment.meaning?.source_keys)}
                    <p className="subtext mt-2">Change being compared</p>
                    {references(treatment.change?.source_keys)}
                  </li>
                ))}
              </ul>
            ) : (
              <span className="subtext">Not selected</span>
            )}
          </dd>
        </div>
        {(["comparison", "population", "unit", "timing"] as const).map(
          (field) => (
            <div key={field}>
              <dt>{field}</dt>
              <dd>{references(brief[field]?.source_keys)}</dd>
            </div>
          ),
        )}
        <div>
          <dt>Explicit assumptions</dt>
          <dd>
            {assumptions.length ? (
              assumptions.map((claim, index) => (
                <div key={index}>{references(claim.source_keys)}</div>
              ))
            ) : (
              <span className="subtext">None recorded in this brief</span>
            )}
          </dd>
        </div>
        <div>
          <dt>Optional unknowns</dt>
          <dd>
            {optionalUnknowns.length ? (
              <ul className="analysis-list">
                {optionalUnknowns.map((unknown, index) => (
                  <li key={index}>{unknown}</li>
                ))}
              </ul>
            ) : (
              <span className="subtext">None recorded</span>
            )}
          </dd>
        </div>
      </dl>
      {"source" in brief && (
        <details className="mt-3">
          <summary>Brief application provenance</summary>
          <pre className="analysis-code">
            {JSON.stringify(brief.source, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
