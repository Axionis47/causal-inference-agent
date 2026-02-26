import { useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { ArrowLeft, BookOpen, GitBranch } from "lucide-react";
import { api, client, errorMessage, queryClient } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { ErrorMessage, Loading } from "@/components/feedback";
import { Review } from "./review";
import { Sources, SourceViewer } from "./sources";
import { label } from "./types";
import type { Citation } from "./types";

const route = "/datasets/$datasetId/context/$versionId" as const;
type Props = {
  datasetId: string;
  versionId: string;
  revision?: string;
  fromConversation?: string;
  section: string;
};

export function ContextWorkspace({
  datasetId,
  versionId,
  revision,
  fromConversation,
  section,
}: Props) {
  const workspace = api.useQuery(
    "get",
    "/api/versions/{version_id}/context",
    { params: { path: { version_id: versionId } } },
    {
      refetchInterval: (query) =>
        query.state.data?.preparations.some((p) =>
          ["queued", "running"].includes(p.status),
        )
          ? 1500
          : 15000,
    },
  );
  const dataset = api.useQuery("get", "/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: datasetId } },
  });
  const currentId = workspace.data?.current_revision_id;
  const manifest = api.useQuery(
    "get",
    "/api/versions/{version_id}/context/revisions/{revision_id}/read",
    {
      params: {
        path: { version_id: versionId, revision_id: currentId ?? "" },
        query: { key: ["sources"] },
      },
    },
    { enabled: !!currentId },
  );
  const [source, setSource] = useState<{
    id: string;
    citation?: Citation;
  } | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const prepareKey = useRef<{ signature: string; id: string } | null>(null);
  const refresh = async () => {
    await queryClient.invalidateQueries({
      predicate: (q) =>
        q.queryKey.some((k) => typeof k === "string" && k.includes("/context")),
    });
  };
  const prepare = useMutation({
    mutationFn: async () => {
      const signature = JSON.stringify([currentId, selected]);
      if (prepareKey.current?.signature !== signature)
        prepareKey.current = { signature, id: crypto.randomUUID() };
      const result = await client.POST(
        "/api/versions/{version_id}/context/preparations",
        {
          params: { path: { version_id: versionId } },
          body: {
            submission_id: prepareKey.current.id,
            expected_revision_id: currentId ?? null,
            source_ids: selected,
          },
        },
      );
      if (result.error) throw new Error(errorMessage(result.error));
      return result.data;
    },
    onSuccess: async () => {
      prepareKey.current = null;
      setSelected([]);
      await refresh();
    },
  });
  const retry = useMutation({
    mutationFn: async (id: string) => {
      const result = await client.POST(
        "/api/versions/{version_id}/context/preparations/{preparation_id}/retry",
        {
          params: { path: { version_id: versionId, preparation_id: id } },
          bodySerializer: () => "{}",
        },
      );
      if (result.error) throw new Error(errorMessage(result.error));
    },
    onSuccess: refresh,
  });
  if (workspace.isPending) return <Loading>Opening context workspace…</Loading>;
  if (workspace.error || !workspace.data)
    return <ErrorMessage error={workspace.error} />;
  const data = workspace.data;
  const included = Object.keys(manifest.data?.sources ?? {});
  const pendingSources = data.sources.filter((s) => !included.includes(s.id));
  const operation = data.preparations[0];
  const busy = data.preparations.some(
    (p) => p.status === "queued" || p.status === "running",
  );
  return (
    <>
      <Link
        to="/datasets/$datasetId"
        params={{ datasetId }}
        search={{ page: 1 }}
        className="inline-flex gap-2 items-center text-muted-foreground"
      >
        <ArrowLeft size={13} aria-hidden="true" />
        Dataset preview
      </Link>
      <div className="page-heading context-page-heading">
        <div>
          <div className="eyebrow">CONTEXT / SOURCE-GROUNDED ACCOUNT</div>
          <h1 className="mt-4">{dataset.data?.title ?? "Dataset"}</h1>
          <p className="subtext">
            Keep the supplied account inspectable. Review what is known; leave
            gaps visible.
          </p>
        </div>
        <BookOpen size={30} className="text-primary" aria-hidden="true" />
      </div>
      <div className="context-toolbar">
        {fromConversation && (
          <Button asChild variant="outline">
            <Link
              to="/datasets/$datasetId/chat/$versionId"
              params={{ datasetId, versionId }}
              search={{ conversation: fromConversation, view: "chat" }}
            >
              Back to conversation
            </Link>
          </Button>
        )}
        <Button asChild variant="outline"><Link to="/datasets/$datasetId/chat/$versionId" params={{ datasetId, versionId }} search={{ revision: revision ?? currentId ?? undefined }}>Open chat with this context</Link></Button>
        <details
          className="context-tool"
          open={!currentId || busy || operation?.status === "failed"}
        >
          <summary>
            <BookOpen size={15} aria-hidden="true" />
            <strong>Sources & preparation</strong>
            <span>
              {data.sources.length}{" "}
              {data.sources.length === 1 ? "source" : "sources"}
            </span>
            <span
              className={`preparation-pill ${operation?.status ?? "empty"}`}
            >
              {busy
                ? "Processing"
                : operation?.status === "succeeded"
                  ? "Draft prepared"
                  : operation?.status === "failed"
                    ? "Needs attention"
                    : "Add sources"}
            </span>
          </summary>
          <div className="context-tool-content">
            <Sources
              versionId={versionId}
              workspace={data}
              included={included}
              refresh={refresh}
              onInspect={(s) => setSource({ id: s.id })}
            />
            <section className="panel mb-6" aria-labelledby="prepare-title">
              <div className="panel-heading">
                <h2 id="prepare-title" className="panel-title">
                  02 / Prepare context
                </h2>
                <span className="eyebrow">
                  {busy ? "PROCESSING" : "SAVED SOURCES"}
                </span>
              </div>
              <div className="p-5">
                <p className="subtext">
                  Select new sources to organize into the original context keys.
                  Reviewed statements are retained.
                </p>
                {pendingSources.length ? (
                  <fieldset
                    className="source-selection"
                    disabled={
                      busy ||
                      prepare.isPending ||
                      (!!currentId && manifest.isPending)
                    }
                  >
                    <legend className="sr-only">Sources to prepare</legend>
                    {pendingSources.map((s) => (
                      <label className="context-check" key={s.id}>
                        <input
                          type="checkbox"
                          checked={selected.includes(s.id)}
                          onChange={(e) =>
                            setSelected((old) =>
                              e.target.checked
                                ? [...old, s.id]
                                : old.filter((id) => id !== s.id),
                            )
                          }
                        />
                        {s.title}
                      </label>
                    ))}
                  </fieldset>
                ) : (
                  <p className="subtext mt-3">
                    {data.sources.length
                      ? "All uploaded sources are in the current revision."
                      : "Upload source material to begin."}
                  </p>
                )}
                <Button
                  className="mt-4"
                  disabled={!selected.length || busy || prepare.isPending}
                  onClick={() => prepare.mutate()}
                >
                  {busy
                    ? "Preparation in progress…"
                    : "Prepare selected sources"}
                </Button>
                {operation && (
                  <div
                    className="preparation-status"
                    role="status"
                    aria-live="polite"
                  >
                    <p>
                      {label(operation.status)} · {label(operation.stage)} ·{" "}
                      {operation.calls} of 4 model calls used
                    </p>
                    {operation.error && (
                      <p className="field-error mt-2">{operation.error}</p>
                    )}
                    {operation.status === "failed" &&
                      operation.calls < 4 &&
                      operation.stage !== "rejected" && (
                        <Button
                          className="mt-3"
                          variant="outline"
                          onClick={() => retry.mutate(operation.id)}
                          disabled={retry.isPending}
                        >
                          Retry preparation
                        </Button>
                      )}
                  </div>
                )}
                {prepare.error && <ErrorMessage error={prepare.error} />}{" "}
                {retry.error && <ErrorMessage error={retry.error} />}{" "}
                {manifest.error && <ErrorMessage error={manifest.error} />}
              </div>
            </section>
          </div>
        </details>
        <details className="context-tool context-history-tool">
          <summary>
            <GitBranch size={15} aria-hidden="true" />
            <strong>History</strong>
            <span>{data.revisions.length} saved</span>
          </summary>
          <section className="panel mb-6" aria-label="Context revision history">
            <div className="panel-heading">
              <h2 className="panel-title">
                <GitBranch size={14} aria-hidden="true" />
                03 / Review revisions
              </h2>
              <span className="eyebrow">{data.revisions.length} SAVED</span>
            </div>
            <div className="revision-list">
              {data.revisions.length ? (
                data.revisions.map((r) => (
                  <Link
                    key={r.id}
                    to={route}
                    params={{ datasetId, versionId }}
                    search={{ revision: r.id, section }}
                    className={r.id === revision ? "tag" : "context-link"}
                    aria-current={r.id === revision ? "page" : undefined}
                  >
                    Revision {r.number} · {r.state}
                    {r.id === currentId ? " · current" : ""}
                  </Link>
                ))
              ) : (
                <p className="subtext">
                  The first successful preparation creates a draft revision
                  here.
                </p>
              )}
            </div>
          </section>
        </details>
      </div>
      {revision || currentId ? (
        <Review
          key={revision ?? currentId}
          datasetId={datasetId}
          versionId={versionId}
          revision={(revision ?? currentId)!}
          section={section}
          workspace={data}
          refresh={refresh}
          onCitation={(cite) =>
            setSource({ id: cite.source_id, citation: cite })
          }
        />
      ) : null}
      {source && (
        <SourceViewer
          versionId={versionId}
          sourceId={source.id}
          citation={source.citation}
          close={() => setSource(null)}
        />
      )}
    </>
  );
}
