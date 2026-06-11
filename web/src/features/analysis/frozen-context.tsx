import type { components } from "@/lib/api/schema";
import { BriefDetails } from "@/features/context/analysis-brief";
import { RecordDetails } from "./inspector";

type Analysis = components["schemas"]["AnalysisView"];

export function FrozenContext({
  data,
  datasetId,
}: {
  data: Analysis;
  datasetId: string;
}) {
  const frozen = data.frozen_context;
  if (!frozen)
    return (
      <p className="subtext mt-2" aria-label="Frozen context availability">
        Frozen brief and readiness receipt unavailable in this saved record.
      </p>
    );
  return (
    <details className="analysis-section" aria-label="Frozen analysis context">
      <summary>Frozen context / exact applied revision</summary>
      <p className="subtext mt-2">
        This analysis keeps the brief and context bound when it started. Later
        context changes do not change this record.
      </p>
      <p className="analysis-hash">
        Context revision: {frozen.context_revision_id ?? "Unavailable"}
        <br />
        Context hash: {frozen.context_hash}
        <br />
        Readiness fingerprint: {frozen.readiness_fingerprint}
        <br />
        Policy: {frozen.policy_version}
      </p>
      <a
        className="context-link mt-3 inline-block"
        href={`/api/analyses/${data.id}/files/frozen-context.json`}
        target="_blank"
        rel="noreferrer"
      >
        Open exact frozen context receipt
      </a>
      <BriefDetails
        brief={frozen.brief}
        location={{
          datasetId,
          versionId: frozen.dataset_version_id,
          revision: frozen.context_revision_id,
        }}
      />
      <details className="mt-3">
        <summary>Frozen readiness receipt and source identity</summary>
        <RecordDetails
          value={{
            dataset_version_id: frozen.dataset_version_id,
            source_sha256: frozen.source_sha256,
            readiness: frozen.readiness,
          }}
        />
      </details>
    </details>
  );
}
