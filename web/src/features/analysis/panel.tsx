import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { api, client, errorMessage, queryClient } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";
import { Button } from "@/components/ui/button";
import { ErrorMessage, Loading } from "@/components/feedback";
import { CausalGraph } from "./causal-graph";
import { AnalysisResults, FileLink } from "./results";
import { AgentInspector, AgentProgress } from "./agents";
import { TaskInspector } from "./tasks";
import { SharedProviderAttempts } from "./provider-attempts";
import { TreatmentBindings } from "./treatment-bindings";
import { ProgramDecision } from "./program-decision";
import { AnalysisStart } from "./readiness";
import { FrozenContext } from "./frozen-context";
import { AuthorityInspector, AssumptionDetails } from "./authority";
import {
  CheckpointInspector,
  ContextPack,
  RecordDetails,
  SavedAnalysisVersion,
} from "./inspector";

const route = "/datasets/$datasetId/chat/$versionId" as const;
type AnalysisAction =
  | { kind: "approve"; body: components["schemas"]["ApproveAnalysis"] }
  | { kind: "reject"; body: components["schemas"]["RejectAnalysis"] }
  | { kind: "revise"; body: components["schemas"]["RejectAnalysis"] }
  | { kind: "retry" };

export function AnalysisPanel({
  conversationId,
  contextRevisionId,
  contextUpdating,
  datasetId,
  versionId,
  selected,
}: {
  conversationId: string;
  contextRevisionId: string | null;
  contextUpdating: boolean;
  datasetId: string;
  versionId: string;
  selected?: string;
}) {
  const navigate = useNavigate();
  const list = api.useQuery(
    "get",
    "/api/conversations/{conversation_id}/analyses",
    { params: { path: { conversation_id: conversationId } } },
    { refetchInterval: 5000 },
  );
  const identity = selected ?? list.data?.[0]?.id;
  const active = list.data?.some(
    (a) =>
      !a.read_only &&
      ["queued", "running", "awaiting_approval"].includes(a.status),
  );
  return (
    <aside className="panel analysis-panel" aria-label="Analysis workspace">
      <div className="panel-heading">
        <h2 className="panel-title">Analysis</h2>
      </div>
      <div className="analysis-body">
        {list.error && <ErrorMessage error={list.error} />}
        {list.data?.length ? (
          <label className="block">
            Saved analysis
            <select
              className="context-select w-full mt-2"
              value={identity}
              onChange={(event) =>
                void navigate({
                  to: route,
                  params: { datasetId, versionId },
                  search: {
                    conversation: conversationId,
                    analysis: event.target.value,
                    view: "analysis",
                  },
                })
              }
            >
              {list.data.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.question.slice(0, 65)} ·{" "}
                  {item.status.replaceAll("_", " ")}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <p className="subtext">
            Prepare an analysis from your reviewed question and context. You can
            inspect the plan before approving execution.
          </p>
        )}
        {identity && (
          <AnalysisDetail
            key={identity}
            identity={identity}
            conversationId={conversationId}
            datasetId={datasetId}
            versionId={versionId}
          />
        )}
        <AnalysisStart
          key={contextRevisionId ?? "no-context"}
          conversationId={conversationId}
          contextRevisionId={contextRevisionId}
          contextUpdating={contextUpdating}
          activity={
            list.isPending
              ? "checking"
              : list.error
                ? "unavailable"
                : active
                  ? "active"
                  : "idle"
          }
          datasetId={datasetId}
          versionId={versionId}
          onPrepared={async (id) => {
            await list.refetch();
            await navigate({
              to: route,
              params: { datasetId, versionId },
              search: {
                conversation: conversationId,
                analysis: id,
                view: "analysis",
              },
            });
          }}
        />
      </div>
    </aside>
  );
}

function AnalysisDetail({
  identity,
  conversationId,
  datasetId,
  versionId,
}: {
  identity: string;
  conversationId: string;
  datasetId: string;
  versionId: string;
}) {
  const analysis = api.useQuery(
    "get",
    "/api/analyses/{run_id}",
    { params: { path: { run_id: identity } } },
    {
      refetchInterval: (query) =>
        !query.state.data?.read_only &&
        ["queued", "running"].includes(query.state.data?.status ?? "")
          ? 10000
          : false,
    },
  );
  const [connection, setConnection] = useState("Connecting to saved events…");
  const [selectedEvidence, setSelectedEvidence] = useState<string | null>(null);
  const evidencePanel = useRef<HTMLDivElement>(null);
  const technicalPanel = useRef<HTMLDetailsElement>(null);
  const busy =
    !analysis.data?.read_only &&
    ["queued", "running"].includes(analysis.data?.status ?? "");
  const cursor = useRef(0);
  const inspect = (key: string) => {
    setSelectedEvidence(key);
    if (technicalPanel.current) technicalPanel.current.open = true;
    requestAnimationFrame(() => {
      evidencePanel.current?.focus({ preventScroll: true });
      evidencePanel.current?.scrollIntoView({
        block: "nearest",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
      });
    });
  };
  useEffect(() => {
    if (!busy) return;
    const stream = new EventSource(
      `/api/analyses/${identity}/events?after=${cursor.current}`,
    );
    const query = {
      queryKey: [
        "get",
        "/api/analyses/{run_id}",
        { params: { path: { run_id: identity } } },
      ],
    };
    let refreshTimer: ReturnType<typeof setTimeout> | undefined;
    let disposed = false;
    const refreshNow = () =>
      queryClient.invalidateQueries(query, { cancelRefetch: false });
    const refresh = () => {
      if (refreshTimer !== undefined) return;
      refreshTimer = setTimeout(() => {
        refreshTimer = undefined;
        void refreshNow();
      }, 100);
    };
    stream.onopen = () => setConnection("Live event stream");
    stream.onerror = () =>
      setConnection(
        "Disconnected, replay will resume from the last saved event",
      );
    stream.addEventListener("checkpoint", (event: MessageEvent) => {
      cursor.current = Math.max(cursor.current, Number(event.lastEventId));
      refresh();
    });
    stream.addEventListener("settled", () => {
      clearTimeout(refreshTimer);
      refreshTimer = undefined;
      // Reuse a pending read, then refresh once more if it could predate settlement.
      const wasFetching = queryClient.isFetching(query) > 0;
      void refreshNow().then(() => {
        if (wasFetching && !disposed) void refreshNow();
      });
      stream.close();
      void queryClient.invalidateQueries(
        {
          queryKey: ["get", "/api/conversations/{conversation_id}/analyses"],
        },
        { cancelRefetch: false },
      );
      void queryClient.invalidateQueries(
        { queryKey: ["get", "/api/conversations/{conversation_id}"] },
        { cancelRefetch: false },
      );
    });
    return () => {
      disposed = true;
      clearTimeout(refreshTimer);
      stream.close();
    };
  }, [busy, identity]);
  const action = useMutation({
    mutationFn: async (input: AnalysisAction) => {
      const data = analysis.data!;
      if (data.read_only)
        throw new Error("This historical analysis is read-only.");
      const result =
        input.kind === "approve"
          ? await client.POST("/api/analyses/{run_id}/approve", {
              params: { path: { run_id: identity } },
              body: input.body,
            })
          : input.kind === "reject"
            ? await client.POST("/api/analyses/{run_id}/reject", {
                params: { path: { run_id: identity } },
                body: input.body,
              })
            : input.kind === "revise"
              ? await client.POST("/api/analyses/{run_id}/revise-program", {
                  params: { path: { run_id: identity } },
                  body: input.body,
                })
              : await client.POST("/api/analyses/{run_id}/retry", {
                  params: { path: { run_id: identity } },
                });
      if (result.error) throw new Error(errorMessage(result.error));
    },
    onSuccess: async () => {
      await Promise.all([
        analysis.refetch(),
        queryClient.invalidateQueries({
          queryKey: ["get", "/api/conversations/{conversation_id}/analyses"],
        }),
        queryClient.invalidateQueries({
          queryKey: ["get", "/api/conversations/{conversation_id}"],
        }),
      ]);
    },
  });
  if (analysis.isPending) return <Loading>Opening saved trajectory…</Loading>;
  if (analysis.error || !analysis.data)
    return <ErrorMessage error={analysis.error} />;
  const data = analysis.data;
  if (data.conversation_id !== conversationId)
    return (
      <ErrorMessage
        error={new Error("This analysis belongs to another conversation.")}
      />
    );
  const assumptions = [
    ...new Map(
      [
        ...(data.snapshot?.frame.causal.assumptions ?? []),
        ...data.checkpoints.flatMap((checkpoint) =>
          checkpoint.attempts
            .filter((a) => a.status === "accepted")
            .flatMap((a) => a.assumptions),
        ),
      ].map((a) => [a.id, a]),
    ).values(),
  ];
  const current = data.programs.at(-1);
  const rejectedProgramHashes = new Set(
    data.checkpoints
      .flatMap((checkpoint) => checkpoint.attempts)
      .filter((receipt) =>
        receipt.checks.some(
          (check) => check.rule === "human_program_review" && !check.passed,
        ),
      )
      .map((receipt) => receipt.output.program_hash)
      .filter((hash): hash is string => typeof hash === "string"),
  );
  const lastEvent = data.events.at(-1);
  const statusLabel: Record<string, string> = {
    queued: "Waiting to start",
    running: "Analysis in progress",
    awaiting_approval: "Ready for your review",
    succeeded: data.result?.execution_ok
      ? "Results ready"
      : "Analysis finished",
    blocked: "Analysis needs attention",
    failed: "Analysis interrupted",
  };
  return (
    <section className="mt-5">
      <p className="analysis-status" role="status">
        {statusLabel[data.status] ?? data.status.replaceAll("_", " ")}
      </p>
      <p className="mt-3 whitespace-pre-wrap">{data.question}</p>
      {data.result && (
        <AnalysisResults
          data={data}
          datasetId={datasetId}
          versionId={versionId}
          conversationId={conversationId}
        />
      )}
      {data.result ? (
        <details className="analysis-section">
          <summary>Completed workflow</summary>
          <AgentProgress data={data} />
        </details>
      ) : (
        <AgentProgress data={data} />
      )}
      {(data.status === "failed" || data.status === "blocked") && (
        <p className="subtext mt-3">
          {data.status === "failed"
            ? "The analysis could not finish. Your saved inputs and progress are preserved."
            : data.outcome === "unidentified"
              ? "The requested effect is not identified under the applied context and stated assumptions."
              : data.outcome === "unsupported"
                ? "The requested analysis is not supported under the applied context and available methods."
                : "The analysis stopped. Its inputs, progress and review records remain saved."}
        </p>
      )}
      {data.outcome && data.error && <p className="mt-2">{data.error}</p>}
      {!data.read_only && data.retryable && (
        <Button
          className="mt-3"
          variant="outline"
          disabled={action.isPending}
          onClick={() => action.mutate({ kind: "retry" })}
        >
          Resume saved work
        </Button>
      )}
      {action.error && <ErrorMessage error={action.error} />}
      <FrozenContext data={data} datasetId={datasetId} />
      {data.read_only && (
        <p className="subtext mt-2">
          This analysis uses a historical contract and is read-only. Its saved
          decisions, code and results remain available. Prepare a new analysis
          to use the current workflow.
        </p>
      )}
      {(data.frame ||
        data.checkpoints.some(
          (checkpoint) =>
            checkpoint.id === "treatments" && checkpoint.status === "accepted",
        )) && (
        <details className="analysis-section">
          <summary>Treatments and comparisons</summary>
          <TreatmentBindings data={data} inspect={inspect} />
        </details>
      )}
      {data.authority && (
        <details className="analysis-section">
          <summary>Assumptions and source support</summary>
          <AuthorityInspector data={data} inspect={inspect} />
        </details>
      )}
      {data.frame && (
        <details className="analysis-section">
          <summary>Causal account</summary>
          <p className="analysis-hash">{data.frame.graph_hash}</p>
          <CausalGraph frame={data.frame} inspect={inspect} />
          <p className="eyebrow mt-3">Proposed identification explanation</p>
          <p className="mt-3">{data.frame.causal.identification}</p>
        </details>
      )}
      {!data.authority && !!assumptions.length && (
        <details className="analysis-section">
          <summary>Analysis assumptions / {assumptions.length}</summary>
          <ul className="analysis-list">
            {assumptions.map((a) => (
              <li key={a.id}>
                <AssumptionDetails assumption={a} inspect={inspect} />
              </li>
            ))}
          </ul>
        </details>
      )}
      {!!data.diagnostics.length && (
        <details className="analysis-section">
          <summary>
            Computational diagnostics / {data.diagnostics.length}
          </summary>
          {data.diagnostics.map((record, index) => (
            <details className="mt-3" key={String(record.reference)}>
              <summary>
                {String(record.reference)}: {String(record.question)}
              </summary>
              <RecordDetails value={record} />
              <FileLink run={identity} name={`diagnostic-${index}.py`}>
                Download diagnostic code
              </FileLink>
            </details>
          ))}
        </details>
      )}
      {data.snapshot && (
        <details className="analysis-section">
          <summary>Analysis method and frozen inputs</summary>
          <p className="analysis-hash">Science hash: {data.snapshot_hash}</p>
          <RecordDetails value={data.snapshot.plan} />
          {data.snapshot.input_projection && (
            <details className="mt-3" aria-label="Frozen execution input">
              <summary>
                Frozen execution input /{" "}
                {data.snapshot.input_projection.columns.length} selected columns
              </summary>
              <p className="subtext mt-2">
                The program receives these columns under their canonical IDs.
                All source rows and selected CSV cell text are preserved.
              </p>
              <p className="analysis-hash">
                Source checksum: {data.snapshot.source_sha256}
              </p>
              <RecordDetails value={data.snapshot.input_projection} />
            </details>
          )}
          <p className="subtext mt-3">{data.snapshot.prior_exposure}</p>
          <FileLink run={identity} name="snapshot.json">
            Download exact snapshot
          </FileLink>
        </details>
      )}
      {!!data.programs.length && (
        <details
          className="analysis-section"
          open={data.status === "awaiting_approval"}
        >
          <summary>Analysis program and review history</summary>
          {data.programs.map((program) => (
            <details
              key={program.hash}
              open={program === current && data.status === "awaiting_approval"}
            >
              <summary>
                Revision {program.revision + 1} ·{" "}
                {program.approved_at
                  ? "approved"
                  : program.rejected_reason ||
                      rejectedProgramHashes.has(program.hash)
                    ? "rejected during review"
                    : data.read_only
                      ? "unapproved historical program"
                      : "awaiting approval"}
              </summary>
              <p className="mt-3">{program.program.summary}</p>
              <p className="subtext mt-2">
                Proposed review explanation: {program.review.summary}
              </p>
              <p className="subtext mt-2">
                {program.verification
                  ? `Deterministic verification: ${program.verification.analyzer} ${program.verification.version}, ${program.verification.rules.join(", ")}. No target data accessed.`
                  : "Deterministic verification: unavailable for this historical program."}
              </p>
              {program.verification && (
                <p className="subtext mt-2">
                  {program.verification.limitations}
                </p>
              )}
              <pre className="analysis-code mt-3">{program.program.code}</pre>
              <FileLink run={identity} name={`program-${program.revision}.py`}>
                Download this program
              </FileLink>
              {(program.rejected_reason ||
                rejectedProgramHashes.has(program.hash)) && (
                <div className="mt-2">
                  <FileLink
                    run={identity}
                    name={`review-rejection-${program.revision}.json`}
                  >
                    Download saved rejection
                  </FileLink>
                </div>
              )}
              {program.diff && (
                <details open className="mt-3">
                  <summary>Changes from the preceding program</summary>
                  <pre className="analysis-code">{program.diff}</pre>
                </details>
              )}
              <p className="analysis-hash">
                Program hash: {program.hash}
                <br />
                Runtime: {program.runtime_hash}
              </p>
              {program.execution && (
                <details className="mt-3">
                  <summary>
                    Preserved execution receipt and input provenance
                  </summary>
                  <RecordDetails value={program.execution} />
                </details>
              )}
            </details>
          ))}
        </details>
      )}
      {!data.read_only &&
        data.status === "awaiting_approval" &&
        current &&
        !current.approved_at &&
        data.snapshot_hash && (
          <ProgramDecision
            key={current.hash}
            revision={current.revision}
            pending={
              action.isPending &&
              action.variables &&
              action.variables.kind !== "retry"
                ? action.variables.kind
                : null
            }
            decide={(decision) => {
              const body = {
                snapshot_hash: data.snapshot_hash!,
                program_hash: current.hash,
              };
              action.mutate(
                decision.kind !== "approve"
                  ? {
                      kind: decision.kind,
                      body: { ...body, reason: decision.reason },
                    }
                  : { kind: "approve", body },
              );
            }}
          />
        )}
      {!data.result && current?.execution?.valid === true && (
        <section className="analysis-section">
          <h3 className="eyebrow">
            Validated computation / report not accepted
          </h3>
          <p className="subtext mt-3">
            The approved program produced validated numerical outputs. Its
            explanation is still being prepared. The saved computation remains
            available for inspection and discussion in chat.
          </p>
          <details className="mt-3">
            <summary>Computed facts</summary>
            <RecordDetails value={current.execution.facts} />
          </details>
          <FileLink
            run={identity}
            name={`execution-${current.revision}-facts.json`}
          >
            Download saved computation
          </FileLink>
        </section>
      )}
      <details
        ref={technicalPanel}
        className="analysis-section"
        aria-label="Analysis technical details"
      >
        <summary>Technical details</summary>
        <p className="subtext mt-3">
          Saved workflow receipts, service errors, and execution logs for
          troubleshooting.
        </p>
        <SavedAnalysisVersion data={data} />
        {data.error && <p className="field-error mt-3">{data.error}</p>}
        <p className="subtext mt-2">
          {data.budgets.model_calls} / {data.budgets.model_calls_limit} model
          calls · {data.budgets.diagnostic_runs} /{" "}
          {data.budgets.diagnostic_runs_limit} diagnostic executions
        </p>
        <p className="subtext mt-2">
          {busy ? connection : "Viewing persisted trajectory"}
          <br />
          Last saved activity:{" "}
          {lastEvent
            ? new Date(lastEvent.created_at).toLocaleString()
            : "Unavailable"}
        </p>
        {data.legacy_record ? (
          <details className="mt-3">
            <summary>Original saved record</summary>
            <pre className="analysis-code">
              {JSON.stringify(data.legacy_record, null, 2)}
            </pre>
          </details>
        ) : (
          <>
            {data.agents?.length ? (
              <AgentInspector data={data} />
            ) : (
              <>
                <TaskInspector data={data} inspect={inspect} />
                <CheckpointInspector data={data} inspect={inspect} />
                <SharedProviderAttempts data={data} />
              </>
            )}
          </>
        )}
        <div
          ref={evidencePanel}
          tabIndex={-1}
          aria-label="Selected analysis evidence"
        >
          <ContextPack
            data={data}
            selected={selectedEvidence}
            select={setSelectedEvidence}
          />
        </div>
        {data.context_revision_id && (
          <div className="flex flex-wrap gap-3 mt-3">
            <Link
              className="context-link"
              target="_blank"
              rel="noreferrer"
              to="/datasets/$datasetId/context/$versionId"
              params={{ datasetId, versionId }}
              search={{
                revision: data.context_revision_id,
                section: "dataset",
              }}
            >
              Open this context revision
            </Link>
            <FileLink run={identity} name="context.json">
              Download context pack
            </FileLink>
          </div>
        )}
        <details className="analysis-section">
          <summary>Terminal / saved reasoning events</summary>
          <p className="subtext mt-2">
            Event text records activity and proposed explanations. Open the
            corresponding receipt for claim authority and gate results.
          </p>
          <div
            className="analysis-terminal"
            role="log"
            aria-label="Analysis checkpoint events"
            aria-live="polite"
            tabIndex={0}
          >
            {data.events.map((event) => (
              <div key={event.sequence}>
                <p>
                  <span>{String(event.sequence).padStart(3, "0")}</span> [
                  {event.kind}] {event.stage}{" "}
                  <time dateTime={event.created_at}>
                    {new Date(event.created_at).toLocaleTimeString()}
                  </time>
                </p>
                <p>{event.summary}</p>
                {Object.keys(event.detail).length > 0 && (
                  <details>
                    <summary>Event receipt</summary>
                    <pre>{JSON.stringify(event.detail, null, 2)}</pre>
                  </details>
                )}
              </div>
            ))}
          </div>
          <FileLink run={identity} name="trajectory.json">
            Download complete trajectory
          </FileLink>
        </details>
      </details>
    </section>
  );
}
