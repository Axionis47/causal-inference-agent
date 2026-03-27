import { type ReactNode, useLayoutEffect, useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Link, useBlocker, useNavigate } from "@tanstack/react-router";
import { useMutation } from "@tanstack/react-query";
import { ArrowLeft, MessageSquare, Plus } from "lucide-react";
import { api, client, errorMessage, queryClient } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ErrorMessage, Loading } from "@/components/feedback";
import { Composer } from "./composer";
import { ArtifactCard } from "./artifact";
import { BriefDetails } from "@/features/context/analysis-brief";

const route = "/datasets/$datasetId/chat/$versionId" as const;
const createSchema = z.object({
  title: z.string().trim().min(1, "Give this conversation a title.").max(200),
  context_revision_id: z.string(),
});
export type ConversationPanelState = {
  conversationId: string;
  contextRevisionId: string | null;
  contextUpdating: boolean;
  datasetId: string;
  versionId: string;
  selected?: string;
};

type Props = {
  datasetId: string;
  versionId: string;
  conversation?: string;
  revision?: string;
  analysis?: string;
  view: "chat" | "analysis";
  renderSidePanel: (state: ConversationPanelState) => ReactNode;
};

export function ChatWorkspace({
  datasetId,
  versionId,
  conversation,
  revision,
  analysis,
  view,
  renderSidePanel,
}: Props) {
  const navigate = useNavigate();
  const dataset = api.useQuery("get", "/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: datasetId } },
  });
  const list = api.useQuery("get", "/api/versions/{version_id}/conversations", {
    params: { path: { version_id: versionId } },
  });
  const context = api.useQuery("get", "/api/versions/{version_id}/context", {
    params: { path: { version_id: versionId } },
  });
  const form = useForm<z.infer<typeof createSchema>>({
    resolver: zodResolver(createSchema),
    defaultValues: { title: "", context_revision_id: revision ?? "" },
  });
  const submission = useRef<{ signature: string; id: string } | null>(null);
  useBlocker({
    shouldBlockFn: () =>
      !conversation &&
      form.formState.isDirty &&
      !window.confirm("Discard the unsaved conversation details?"),
    enableBeforeUnload: !conversation && form.formState.isDirty,
  });
  const create = useMutation({
    mutationFn: async (values: z.infer<typeof createSchema>) => {
      const signature = JSON.stringify(values);
      if (submission.current?.signature !== signature)
        submission.current = { signature, id: crypto.randomUUID() };
      const result = await client.POST(
        "/api/versions/{version_id}/conversations",
        {
          params: { path: { version_id: versionId } },
          body: {
            ...values,
            context_revision_id: values.context_revision_id || null,
            submission_id: submission.current.id,
          },
        },
      );
      if (result.error) throw new Error(errorMessage(result.error));
      return result.data;
    },
    onSuccess: async (data) => {
      submission.current = null;
      form.reset();
      await list.refetch();
      await navigate({
        to: route,
        params: { datasetId, versionId },
        search: { conversation: data.id, view },
      });
    },
  });
  return (
    <>
      <Link
        to="/datasets/$datasetId"
        params={{ datasetId }}
        search={{ page: 1, fromConversation: conversation }}
        className="inline-flex items-center gap-2 text-muted-foreground"
      >
        <ArrowLeft size={13} aria-hidden="true" />
        Dataset preview
      </Link>
      <div className="page-heading chat-page-heading">
        <div>
          <div className="eyebrow mt-6">
            {view === "analysis"
              ? "ANALYSIS / QUESTION TO RESULTS"
              : "CHAT / EXPLORE & DISCUSS"}
          </div>
          <h1 className="mt-3">
            {dataset.data?.title ?? "Dataset conversation"}
          </h1>
          <p className="subtext">
            {view === "analysis"
              ? "Prepare, approve, and inspect results from your applied context."
              : "Ask, visualize, and follow the evidence across turns."}
          </p>
        </div>
        <MessageSquare className="text-primary" aria-hidden="true" />
      </div>
      <div className="chat-layout">
        <aside className="panel chat-list" aria-label="Saved conversations">
          <div className="panel-heading">
            <h2 className="panel-title">Conversations</h2>
            <Button asChild variant="ghost" size="icon-sm">
              <Link
                to={route}
                params={{ datasetId, versionId }}
                search={{ revision, view }}
                aria-label="New conversation"
              >
                <Plus size={15} />
              </Link>
            </Button>
          </div>
          {list.isPending ? (
            <Loading />
          ) : list.error ? (
            <ErrorMessage error={list.error} />
          ) : list.data?.length ? (
            <nav>
              {list.data.map((item) => (
                <Link
                  className="chat-list-item"
                  key={item.id}
                  to={route}
                  params={{ datasetId, versionId }}
                  search={{ conversation: item.id, view }}
                  aria-current={item.id === conversation ? "page" : undefined}
                >
                  {item.title}
                  <span>
                    {new Date(item.created_at).toLocaleString(undefined, {
                      dateStyle: "medium",
                      timeStyle: "short",
                    })}
                  </span>
                </Link>
              ))}
            </nav>
          ) : (
            <p className="p-5 subtext">No saved chats yet.</p>
          )}
        </aside>
        {conversation ? (
          <Conversation
            key={conversation}
            id={conversation}
            datasetId={datasetId}
            versionId={versionId}
            analysis={analysis}
            view={view}
            renderSidePanel={renderSidePanel}
          />
        ) : (
          <section className="panel p-6">
            <h2 className="text-lg">
              {view === "analysis"
                ? "Choose a conversation for analysis"
                : "Start a conversation"}
            </h2>
            <p className="subtext mt-2">
              {view === "analysis"
                ? "Open a saved conversation, or create one with your applied context to prepare an analysis."
                : "Choose the saved context to carry into this chat. You can leave unknowns unresolved."}
            </p>
            <form
              className="mt-6 space-y-5"
              onSubmit={form.handleSubmit((values) => create.mutate(values))}
            >
              <div>
                <label htmlFor="chat-title">Conversation title</label>
                <Input
                  id="chat-title"
                  className="mt-2"
                  placeholder="What would you like to understand?…"
                  autoComplete="off"
                  aria-invalid={!!form.formState.errors.title}
                  aria-describedby={
                    form.formState.errors.title ? "chat-title-error" : undefined
                  }
                  {...form.register("title")}
                />
                {form.formState.errors.title && (
                  <p id="chat-title-error" className="field-error">
                    {form.formState.errors.title.message}
                  </p>
                )}
              </div>
              <div>
                <label htmlFor="chat-context">Context revision</label>
                <select
                  id="chat-context"
                  className="context-select mt-2 w-full"
                  {...form.register("context_revision_id")}
                >
                  <option value="">Dataset only, no context selected</option>
                  {context.data?.revisions.map((r) => (
                    <option key={r.id} value={r.id}>
                      Revision {r.number} · {r.state}
                      {r.id === context.data.current_revision_id
                        ? " · current"
                        : ""}
                    </option>
                  ))}
                </select>
              </div>
              {context.error && <ErrorMessage error={context.error} />}
              {create.error && <ErrorMessage error={create.error} />}
              <Button
                disabled={create.isPending || context.isPending}
                type="submit"
              >
                {create.isPending ? "Creating…" : "Create conversation"}
              </Button>
            </form>
            <p className="subtext mt-6">
              The visualization agent can write Python for plots and descriptive
              computations. The analysis panel prepares a frozen causal plan for
              your approval.
            </p>
          </section>
        )}
      </div>
    </>
  );
}

function Conversation({
  id,
  datasetId,
  versionId,
  analysis,
  view,
  renderSidePanel,
}: {
  id: string;
  datasetId: string;
  versionId: string;
  analysis?: string;
  view: "chat" | "analysis";
  renderSidePanel: (state: ConversationPanelState) => ReactNode;
}) {
  const [selected, setSelected] = useState<string[]>([]);
  const transcript = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const chat = api.useQuery(
    "get",
    "/api/conversations/{conversation_id}",
    { params: { path: { conversation_id: id } } },
    {
      refetchInterval: (query) =>
        query.state.data?.turns.some(
          (t) => t.status === "queued" || t.status === "running",
        )
          ? 1500
          : 5000,
    },
  );
  const retry = useMutation({
    mutationFn: async (turnId: string) => {
      const result = await client.POST(
        "/api/conversations/{conversation_id}/turns/{turn_id}/retry",
        {
          params: { path: { conversation_id: id, turn_id: turnId } },
          bodySerializer: () => "{}",
        },
      );
      if (result.error) throw new Error(errorMessage(result.error));
    },
    onSuccess: async () => {
      await chat.refetch();
    },
  });
  const apply = useMutation({
    mutationFn: async (turnId: string) => {
      const result = await client.POST(
        "/api/conversations/{conversation_id}/turns/{turn_id}/apply-context",
        { params: { path: { conversation_id: id, turn_id: turnId } } },
      );
      if (result.error) throw new Error(errorMessage(result.error));
    },
    onSuccess: async () => {
      await Promise.all([
        chat.refetch(),
        queryClient.invalidateQueries({
          queryKey: [
            "get",
            "/api/conversations/{conversation_id}/analysis-readiness",
            { params: { path: { conversation_id: id } } },
          ],
        }),
        queryClient.invalidateQueries({
          queryKey: [
            "get",
            "/api/versions/{version_id}/context",
            { params: { path: { version_id: versionId } } },
          ],
        }),
      ]);
    },
  });
  const latestTurn = chat.data?.turns.at(-1);
  useLayoutEffect(() => {
    if (followLatest.current && transcript.current)
      transcript.current.scrollTop = transcript.current.scrollHeight;
  }, [latestTurn?.id, latestTurn?.reply, latestTurn?.stage]);
  if (chat.isPending) return <Loading>Opening saved conversation…</Loading>;
  if (chat.error || !chat.data)
    return (
      <div>
        <ErrorMessage error={chat.error} />
        <Button onClick={() => void chat.refetch()}>Retry conversation</Button>
      </div>
    );
  const data = chat.data;
  if (data.version_id !== versionId)
    return (
      <ErrorMessage
        error={
          new Error("This conversation belongs to another dataset version.")
        }
      />
    );
  const busy = data.turns.some(
    (t) => t.status === "queued" || t.status === "running",
  );
  return (
    <div className="conversation-workspace">
      <section
        className="panel chat-thread"
        aria-label="Conversation"
        hidden={view !== "chat"}
      >
        <div className="panel-heading flex-wrap gap-3">
          <h2 className="panel-title">{data.title}</h2>
          {data.context_revision_id ? (
            <Link
              className="context-link"
              to="/datasets/$datasetId/context/$versionId"
              params={{ datasetId, versionId }}
              search={{
                revision: data.context_revision_id,
                section: "dataset",
                fromConversation: id,
              }}
            >
              Inspect selected context
            </Link>
          ) : (
            <span className="eyebrow">Dataset only</span>
          )}
        </div>
        {data.newer_context_available && (
          <p className="chat-notice">
            A newer context revision exists. This chat keeps its selected
            revision. Start a new chat to use the newer context.
          </p>
        )}
        <div
          className="chat-transcript"
          ref={transcript}
          onScroll={(event) => {
            const node = event.currentTarget;
            followLatest.current =
              node.scrollHeight - node.scrollTop - node.clientHeight < 80;
          }}
          onLoadCapture={() => {
            if (followLatest.current && transcript.current)
              transcript.current.scrollTop = transcript.current.scrollHeight;
          }}
          tabIndex={0}
          role="region"
          aria-label="Conversation messages"
        >
          {!data.turns.length && (
            <div className="chat-empty">
              <MessageSquare size={25} aria-hidden="true" />
              <h3>Begin with a question</h3>
              <p>
                Discuss the supplied account, ask for a plot, or explore a
                comparison. Follow-up messages retain this conversation.
              </p>
            </div>
          )}
          {data.turns.map((turn) => (
            <article className="chat-turn" key={turn.id}>
              <div className="chat-user">
                <div className="eyebrow">
                  YOU / {turn.number.toString().padStart(2, "0")}
                  {turn.mode === "context" ? " / UPDATE CONTEXT" : ""}
                </div>
                <p className="whitespace-pre-wrap mt-3">{turn.text}</p>
              </div>
              <div className="chat-assistant">
                <div className="eyebrow">ASSISTANT</div>
                {turn.reply && (
                  <p className="whitespace-pre-wrap mt-3">{turn.reply.text}</p>
                )}
                {turn.status === "queued" || turn.status === "running" ? (
                  <p role="status" className="chat-progress">
                    {turn.status === "queued"
                      ? "Queued for the worker…"
                      : `${turn.stage.startsWith("action") || turn.stage.startsWith("finish") ? "Exploring data" : turn.stage === "reply" ? "Preparing reply" : "Reading your request"}…`}{" "}
                    <span>{turn.calls} model calls</span>
                  </p>
                ) : null}
                {turn.error && (
                  <div className="mt-3">
                    <p role="alert" className="field-error">
                      {turn.error}
                    </p>
                    {turn.retryable && (
                      <Button
                        className="mt-3"
                        variant="outline"
                        disabled={retry.isPending}
                        onClick={() => retry.mutate(turn.id)}
                      >
                        Retry this turn
                      </Button>
                    )}
                  </div>
                )}
                {turn.reply?.context_keys?.length ? (
                  <Link
                    className="context-link inline-block mt-3"
                    to="/datasets/$datasetId/context/$versionId"
                    params={{ datasetId, versionId }}
                    search={{
                      revision: turn.context_revision_id ?? undefined,
                      fromConversation: id,
                      section: turn.reply.context_keys[0]?.startsWith(
                        "columns.",
                      )
                        ? turn.reply.context_keys[0]
                            ?.split(".")
                            .slice(0, 2)
                            .join(".")
                        : "dataset",
                    }}
                  >
                    View supporting context
                  </Link>
                ) : null}
                {[
                  ...new Set(
                    (turn.reply?.analysis_references ?? []).map(
                      (reference) => reference.split(".")[1],
                    ),
                  ),
                ].map((analysisId) => (
                  <Link
                    key={analysisId}
                    className="context-link block mt-3"
                    to="/datasets/$datasetId/chat/$versionId"
                    params={{ datasetId, versionId }}
                    search={{ conversation: id, analysis: analysisId }}
                  >
                    Inspect supporting analysis {analysisId?.slice(0, 8)}
                  </Link>
                ))}
                {turn.reply?.context_edits?.length ||
                turn.reply?.analysis_brief ? (
                  <section
                    className="chat-artifact"
                    aria-label="Proposed context changes"
                  >
                    <h3 className="panel-title">
                      {turn.applied_revision_id
                        ? "Applied context changes"
                        : "Proposed context changes"}
                    </h3>
                    <ul className="analysis-list">
                      {(turn.reply.context_edits ?? []).map((edit, index) => (
                        <li
                          key={index}
                          id={`context-change-${turn.id}-${index}`}
                          className="scroll-mt-4"
                        >
                          <strong>
                            {edit.action}: {edit.key}
                          </strong>
                          <p>{edit.statement?.value ?? edit.reason}</p>
                          {edit.statement?.qualifications && (
                            <p className="subtext">
                              {edit.statement.qualifications}
                            </p>
                          )}
                        </li>
                      ))}
                    </ul>
                    {turn.reply.analysis_brief && (
                      <details open className="mt-3">
                        <summary>
                          {turn.applied_revision_id
                            ? "Analysis brief applied with this revision"
                            : "Proposed analysis brief"}
                        </summary>
                        <p className="subtext mt-2">
                          {turn.applied_revision_id
                            ? "This is the saved proposal. Open the applied revision for its exact resolved references."
                            : "Apply changes saves this brief and its context changes together. This proposal does not authorize analysis."}
                        </p>
                        <BriefDetails
                          brief={turn.reply.analysis_brief}
                          proposalId={turn.id}
                          location={{
                            datasetId,
                            versionId,
                            revision: turn.context_revision_id ?? null,
                          }}
                        />
                      </details>
                    )}
                    {turn.applied_revision_id ? (
                      <Link
                        className="context-link"
                        to="/datasets/$datasetId/context/$versionId"
                        params={{ datasetId, versionId }}
                        search={{
                          revision: turn.applied_revision_id,
                          section: "dataset",
                        }}
                      >
                        Inspect saved revision
                      </Link>
                    ) : (
                      <Button
                        className="mt-3"
                        disabled={
                          apply.isPending ||
                          busy ||
                          turn.context_revision_id !== data.context_revision_id
                        }
                        onClick={() => apply.mutate(turn.id)}
                      >
                        {apply.isPending ? "Applying…" : "Apply changes"}
                      </Button>
                    )}
                    {!turn.applied_revision_id &&
                      turn.context_revision_id !== data.context_revision_id && (
                        <p className="subtext mt-2">
                          The context changed. Discuss a fresh proposal before
                          applying.
                        </p>
                      )}
                  </section>
                ) : null}
                {data.artifacts
                  .filter((a) => a.turn_id === turn.id)
                  .map((artifact) => (
                    <ArtifactCard
                      key={artifact.id}
                      artifact={artifact}
                      conversationId={id}
                      selected={selected.includes(artifact.id)}
                      onSelect={() =>
                        setSelected((old) =>
                          old.includes(artifact.id)
                            ? old.filter((a) => a !== artifact.id)
                            : old.length < 4
                              ? [...old, artifact.id]
                              : old,
                        )
                      }
                    />
                  ))}
                {turn.script_attempts.length > 0 && (
                  <details className="chat-attempts mt-4">
                    <summary className="cursor-pointer">
                      Visualization attempts ({turn.script_attempts.length})
                    </summary>
                    <ol className="mt-3 space-y-3">
                      {turn.script_attempts.map((attempt) => (
                        <li key={attempt.step}>
                          <strong>
                            Script {attempt.step}:{" "}
                            {attempt.ok ? "succeeded" : "failed"}
                          </strong>
                          {attempt.stderr && (
                            <pre className="whitespace-pre-wrap break-words text-xs mt-2">
                              {attempt.stderr}
                            </pre>
                          )}
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
              </div>
            </article>
          ))}
        </div>
        {retry.error && <ErrorMessage error={retry.error} />}
        {apply.error && <ErrorMessage error={apply.error} />}
        <Composer
          conversationId={id}
          latest={data.turns.at(-1)?.number ?? 0}
          busy={busy}
          selected={selected}
          onSent={async () => {
            followLatest.current = true;
            setSelected([]);
            await chat.refetch();
          }}
        />
      </section>
      <div hidden={view !== "analysis"}>
        {renderSidePanel({
          conversationId: id,
          contextRevisionId: data.context_revision_id,
          contextUpdating: apply.isPending,
          datasetId,
          versionId,
          selected: analysis,
        })}
      </div>
    </div>
  );
}
