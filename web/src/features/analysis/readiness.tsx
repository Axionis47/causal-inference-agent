import { useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { components } from "@/lib/api/schema";
import { api, client, errorMessage, queryClient } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { ErrorMessage, Loading } from "@/components/feedback";
import {
  BriefDetails,
  ContextReferences,
  type ContextLocation,
} from "@/features/context/analysis-brief";
import { RecordDetails } from "./inspector";

type Readiness = components["schemas"]["AnalysisReadiness"];

const labels: Record<Readiness["status"], string> = {
  needs_context: "Needs context",
  ready_with_assumptions: "Ready with assumptions",
  ready: "Ready",
};

export function ReadinessSummary({
  readiness,
  location,
}: {
  readiness: Readiness;
  location: ContextLocation;
}) {
  const sourceLocation = {
    ...location,
    versionId: readiness.dataset_version_id,
    revision: readiness.context_revision_id,
  };
  return (
    <div aria-label="Analysis readiness">
      <p className="analysis-status" role="status">
        {labels[readiness.status]}
      </p>
      {readiness.missing.length > 0 && (
        <ul className="analysis-list">
          {readiness.missing.map((gap, index) => (
            <li key={`${gap.field}:${index}`}>
              <strong>{gap.field.replaceAll("_", " ")}</strong>
              <p>{gap.detail}</p>
              {(gap.source_keys ?? []).length > 0 ? (
                <ContextReferences
                  references={gap.source_keys ?? []}
                  location={sourceLocation}
                />
              ) : (
                <Link
                  className="context-link"
                  to="/datasets/$datasetId/context/$versionId"
                  params={{
                    datasetId: sourceLocation.datasetId,
                    versionId: sourceLocation.versionId,
                  }}
                  search={{
                    revision: sourceLocation.revision ?? undefined,
                    section: "dataset",
                  }}
                >
                  Review context for this gap
                </Link>
              )}
            </li>
          ))}
        </ul>
      )}
      {readiness.uncertainties.length > 0 && (
        <details open className="mt-3">
          <summary>Recorded assumptions and uncertainties</summary>
          {readiness.uncertainties.map((claim, index) => (
            <ContextReferences
              key={index}
              references={claim.source_keys ?? []}
              location={sourceLocation}
            />
          ))}
        </details>
      )}
      {readiness.optional_unknowns.length > 0 && (
        <details className="mt-3">
          <summary>
            Optional unknowns / {readiness.optional_unknowns.length}
          </summary>
          <ul className="analysis-list">
            {readiness.optional_unknowns.map((unknown, index) => (
              <li key={index}>{unknown}</li>
            ))}
          </ul>
        </details>
      )}
      <p className="subtext mt-3">
        Readiness checks the required context. It does not establish causal
        identification or the truth of an assumption.
      </p>
      <details className="mt-3">
        <summary>Readiness checks and exact input identity</summary>
        <RecordDetails value={readiness} />
      </details>
    </div>
  );
}

export function AnalysisStart({
  conversationId,
  contextRevisionId,
  contextUpdating,
  activity,
  datasetId,
  versionId,
  onPrepared,
}: {
  conversationId: string;
  contextRevisionId: string | null;
  contextUpdating: boolean;
  activity: "checking" | "active" | "idle" | "unavailable";
  datasetId: string;
  versionId: string;
  onPrepared: (identity: string) => Promise<void>;
}) {
  const params = { params: { path: { conversation_id: conversationId } } };
  const readiness = api.useQuery(
    "get",
    "/api/conversations/{conversation_id}/analysis-readiness",
    params,
    { refetchOnMount: "always" },
  );
  const brief = api.useQuery(
    "get",
    "/api/versions/{version_id}/context/revisions/{revision_id}/read",
    {
      params: {
        path: {
          version_id: versionId,
          revision_id: contextRevisionId ?? "",
        },
        query: { key: ["analysis_brief"] },
      },
    },
    { enabled: Boolean(contextRevisionId) },
  );
  const submission = useRef<{ signature: string; id: string } | null>(null);
  const prepare = useMutation({
    mutationFn: async (checked: Readiness) => {
      const signature = JSON.stringify([
        contextRevisionId,
        checked.fingerprint,
      ]);
      if (submission.current?.signature !== signature)
        submission.current = { signature, id: crypto.randomUUID() };
      const result = await client.POST(
        "/api/conversations/{conversation_id}/analyses",
        {
          ...params,
          body: {
            submission_id: submission.current.id,
            expected_context_revision_id: contextRevisionId,
            expected_readiness_fingerprint: checked.fingerprint,
          },
        },
      );
      if (result.error) {
        if (result.response.status === 409) {
          submission.current = null;
          await queryClient.invalidateQueries({
            queryKey: [
              "get",
              "/api/conversations/{conversation_id}/analysis-readiness",
              params,
            ],
          });
        }
        throw new Error(errorMessage(result.error));
      }
      return result.data;
    },
    onSuccess: async (data) => {
      submission.current = null;
      await onPrepared(data.id);
    },
  });
  const data = readiness.data;
  const applied = brief.data?.content.analysis_brief;
  const location = { datasetId, versionId, revision: contextRevisionId };
  const sameRevision =
    data?.context_revision_id === contextRevisionId &&
    data?.dataset_version_id === versionId;
  const canRun =
    data &&
    data.status !== "needs_context" &&
    sameRevision &&
    applied &&
    !readiness.error &&
    !brief.error &&
    !readiness.isFetching &&
    !contextUpdating &&
    !prepare.isPending &&
    activity === "idle";
  return (
    <section className="analysis-section" aria-label="Run from applied context">
      <h3 className="eyebrow">Applied analysis brief</h3>
      {readiness.isPending ? (
        <Loading>Checking the selected context…</Loading>
      ) : readiness.error ? (
        <ErrorMessage error={readiness.error} />
      ) : data ? (
        <ReadinessSummary readiness={data} location={location} />
      ) : null}
      {contextUpdating || (data && !sameRevision) ? (
        <p className="subtext mt-3" role="status">
          The selected context is changing. Refresh readiness before running.
        </p>
      ) : null}
      {brief.error && <ErrorMessage error={brief.error} />}
      {contextRevisionId && brief.isPending ? (
        <Loading>Reading the applied brief…</Loading>
      ) : applied ? (
        <details className="mt-3">
          <summary>
            Inspect the applied question and supporting statements
          </summary>
          <BriefDetails
            brief={applied}
            location={location}
            content={brief.data?.content}
          />
        </details>
      ) : (
        <p className="subtext mt-3">
          No applied analysis brief is available. Discuss the question in Update
          context, then Apply changes to save the brief.
        </p>
      )}
      <Link
        className="context-link mt-3 inline-block"
        to="/datasets/$datasetId/context/$versionId"
        params={{ datasetId, versionId }}
        search={{
          revision: contextRevisionId ?? undefined,
          section: "dataset",
        }}
      >
        Review selected context
      </Link>
      {prepare.error && <ErrorMessage error={prepare.error} />}
      <div className="flex flex-wrap gap-3 mt-4">
        <Button
          type="button"
          disabled={!canRun}
          onClick={() => {
            if (canRun && data) prepare.mutate(data);
          }}
        >
          {prepare.isPending ? "Starting…" : "Run analysis"}
        </Button>
        <Button
          type="button"
          variant="outline"
          disabled={
            readiness.isFetching || contextUpdating || prepare.isPending
          }
          onClick={() => void readiness.refetch()}
        >
          {readiness.isFetching ? "Checking…" : "Refresh readiness"}
        </Button>
      </div>
      <p className="subtext mt-3">
        {activity === "active"
          ? "A saved analysis is active. Follow its progress above."
          : activity === "checking"
            ? "Checking saved analyses before starting another…"
            : activity === "unavailable"
              ? "Saved analyses could not be checked. Reload the workspace before starting another."
              : "Run analysis uses this applied brief and context as the agreed input. No context is added during the run. Execution requires approval of the exact plan and program."}
      </p>
    </section>
  );
}
