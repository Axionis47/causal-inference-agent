import { useRef, useState } from "react";
import { useFieldArray, useForm } from "react-hook-form";
import { useMutation } from "@tanstack/react-query";
import { useBlocker, useNavigate } from "@tanstack/react-router";
import { api, client, errorMessage } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { ErrorMessage, Loading } from "@/components/feedback";
import { ClaimEditor } from "./claim-editor";
import { StatementCard } from "./statement-card";
import { ColumnReview } from "./column-review";
import { ContextNavigation } from "./navigation";
import { BriefDetails } from "./analysis-brief";
import { ContextTile, ReviewCounts, ReviewInspector } from "./overview";
import type { Inspection } from "./overview";
import { entries, entrySection, label } from "./types";
import type { Citation, Edit, Entry, Workspace } from "./types";
const route = "/datasets/$datasetId/context/$versionId" as const;

export function Review({
  datasetId,
  versionId,
  revision,
  section,
  workspace,
  refresh,
  onCitation,
}: {
  datasetId: string;
  versionId: string;
  revision: string;
  section: string;
  workspace: Workspace;
  refresh: () => Promise<void>;
  onCitation: (cite: Citation) => void;
}) {
  const navigate = useNavigate();
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const form = useForm<{ edits: Edit[]; leaveRemaining: boolean }>({
    defaultValues: { edits: [], leaveRemaining: false },
  });
  const changes = useFieldArray({ control: form.control, name: "edits" });
  const edits = form.watch("edits");
  const leaveRemaining = form.watch("leaveRemaining");
  const permitNavigation = useRef(false);
  const dirty = edits.length > 0 || leaveRemaining;
  useBlocker({
    shouldBlockFn: ({ next }) =>
      !permitNavigation.current &&
      dirty &&
      (next.pathname !== `/datasets/${datasetId}/context/${versionId}` ||
        !("revision" in next.search) ||
        next.search.revision !== revision) &&
      !window.confirm("Discard unsaved context changes?"),
    enableBeforeUnload: dirty,
  });
  const [editor, setEditor] = useState<{
    section: string;
    entry?: Entry;
  } | null>(null);
  const read = api.useQuery(
    "get",
    "/api/versions/{version_id}/context/revisions/{revision_id}/read",
    {
      params: {
        path: { version_id: versionId, revision_id: revision },
        query: {
          key: [
            section,
            "concepts",
            "sources",
            ...(section === "dataset" ? ["analysis_brief"] : []),
          ],
        },
      },
    },
  );
  const metadata = workspace.revisions.find((r) => r.id === revision);
  const isCurrent = workspace.current_revision_id === revision;
  const submission = useRef<{ signature: string; id: string } | null>(null);
  const save = useMutation({
    mutationFn: async (state: "draft" | "reviewed") => {
      const payload = {
        expected_revision_id: revision,
        state,
        edits,
        leave_remaining_unresolved: leaveRemaining,
      };
      const signature = JSON.stringify(payload);
      if (submission.current?.signature !== signature)
        submission.current = { signature, id: crypto.randomUUID() };
      const result = await client.POST(
        "/api/versions/{version_id}/context/revisions",
        {
          params: { path: { version_id: versionId } },
          body: { ...payload, submission_id: submission.current.id },
        },
      );
      if (result.error) throw new Error(errorMessage(result.error));
      return result.data;
    },
    onSuccess: async (result) => {
      permitNavigation.current = true;
      await refresh();
      await navigate({
        to: route,
        params: { datasetId, versionId },
        search: { revision: result.id, section },
      });
    },
  });
  function stage(edit: Edit) {
    const index = edits.findIndex(
      (e) => e.key === edit.key && e.action !== "add",
    );
    if (index >= 0 && edit.action !== "add") changes.update(index, edit);
    else changes.append(edit);
  }
  const available = read.data
    ? entries(read.data.content, section, workspace)
    : [];
  const current = available.filter((e) => e.claim.status !== "withdrawn");
  const history = available.filter((e) => e.claim.status === "withdrawn");
  const fields =
    section === "dataset"
      ? workspace.dataset_fields
      : section.startsWith("columns.")
        ? workspace.column_fields
        : [];
  const heading = section.startsWith("columns.")
    ? workspace.columns.find((c) => c.column_id === section.split(".")[1])?.name
    : label(section);
  function preview(entry: Entry): Entry {
    const edit = edits.find((e) => e.key === entry.key && e.action !== "add");
    if (!edit) return entry;
    const status =
      edit.action === "withdraw"
        ? "withdrawn"
        : edit.action === "unresolved"
          ? "unresolved"
          : "reviewed";
    return {
      ...entry,
      claim: { ...entry.claim, ...edit.statement, status } as Entry["claim"],
    };
  }
  function card(entry: Entry) {
    return (
      read.data && (
        <StatementCard
          key={entry.key}
          entry={entry}
          pending={edits.find((e) => e.key === entry.key && e.action !== "add")}
          workspace={workspace}
          read={read.data}
          navigation={{ datasetId, versionId, revision }}
          onCitation={onCitation}
          onNavigate={() => setInspection(null)}
          stage={isCurrent ? stage : undefined}
          correct={() =>
            setEditor({
              section: entrySection(entry.key),
              entry: preview(entry),
            })
          }
          undo={() =>
            changes.remove(
              edits.findIndex((e) => e.key === entry.key && e.action !== "add"),
            )
          }
          disabled={save.isPending}
        />
      )
    );
  }
  return (
    <div className="context-review">
      <ContextNavigation
        datasetId={datasetId}
        versionId={versionId}
        revision={revision}
        section={section}
        columns={workspace.columns}
        onNavigate={() => setInspection(null)}
      />
      <section
        className="panel context-detail"
        aria-labelledby="context-section"
      >
        <div className="panel-heading">
          <div>
            <h2 id="context-section" className="panel-title">
              {heading}
            </h2>
            <p className="subtext">
              {section.startsWith("columns.")
                ? "Meaning, measurement and connections"
                : "Choose a card to inspect its claims"}
            </p>
          </div>
          {isCurrent && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => setEditor({ section })}
              disabled={save.isPending || !read.data}
            >
              Add statement
            </Button>
          )}
        </div>
        {!isCurrent && (
          <p className="p-5 subtext">
            Historical revision. Open the current revision to make changes.
          </p>
        )}
        {read.isPending ? (
          <Loading>Reading saved context…</Loading>
        ) : read.error ? (
          <div className="p-5">
            <ErrorMessage error={read.error} />
            <Button onClick={() => void read.refetch()} variant="outline">
              Retry context
            </Button>
          </div>
        ) : (
          <div className="claim-list">
            {section === "dataset" && (
              <details className="analysis-section">
                <summary>Applied analysis brief</summary>
                {read.data?.content.analysis_brief ? (
                  <BriefDetails
                    brief={read.data.content.analysis_brief}
                    content={read.data.content}
                    location={{ datasetId, versionId, revision }}
                  />
                ) : (
                  <p className="subtext mt-3">
                    No applied analysis brief is saved in this revision. A
                    historical question is not reconstructed as a brief.
                  </p>
                )}
              </details>
            )}
            {section.startsWith("columns.") && read.data ? (
              <ColumnReview
                key={section}
                columnId={section.split(".")[1]!}
                read={read.data}
                workspace={workspace}
                inspect={setInspection}
                preview={preview}
                navigation={{ datasetId, versionId, revision }}
                addRelationship={
                  isCurrent
                    ? () => setEditor({ section: "relationships" })
                    : undefined
                }
              />
            ) : (
              <>
                <div className="column-summary">
                  <span className="eyebrow">{label(section)} CONTEXT</span>
                  <ReviewCounts entries={current.map(preview)} />
                </div>
                <div className="context-tile-grid general-context-grid">
                  {fields.length
                    ? fields.map((field) => {
                        const claims = current.filter(
                          (e) => e.key.split(".").at(-2) === field,
                        );
                        return (
                          <ContextTile
                            key={field}
                            title={label(field)}
                            entries={claims.map(preview)}
                            onOpen={() =>
                              setInspection({
                                title: label(field),
                                entries: claims,
                                emptyFields: claims.length ? [] : [field],
                              })
                            }
                          />
                        );
                      })
                    : current.map((entry, index) => (
                        <ContextTile
                          key={entry.key}
                          title={
                            section === "relationships"
                              ? `Relationship ${index + 1}`
                              : label(
                                  "kind" in entry.claim
                                    ? (entry.claim.kind ?? "Concept")
                                    : "Concept",
                                )
                          }
                          entries={[preview(entry)]}
                          onOpen={() =>
                            setInspection({
                              title:
                                section === "relationships"
                                  ? "Relationship details"
                                  : "Concept details",
                              entries: [entry],
                            })
                          }
                        />
                      ))}
                </div>
                {!current.length && !fields.length && (
                  <p className="subtext py-5">
                    No current statements supplied for this section.
                  </p>
                )}
              </>
            )}
            {history.length > 0 && (
              <Button
                variant="ghost"
                className="mt-4"
                onClick={() =>
                  setInspection({
                    title: "Withdrawn statements",
                    entries: history,
                  })
                }
              >
                {history.length} withdrawn statements
              </Button>
            )}
          </div>
        )}
        {edits.some((e) => e.action === "add") && (
          <div className="p-5 border-t border-border">
            <h3 className="panel-title">Unsaved additions</h3>
            {edits.map((edit, index) =>
              edit.action === "add" ? (
                <div key={index} className="pending-addition">
                  <span>
                    {edit.statement?.value}
                    <small className="context-key">{edit.key}</small>
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => changes.remove(index)}
                  >
                    Remove change
                  </Button>
                </div>
              ) : null,
            )}
          </div>
        )}
        {isCurrent && (
          <div className="context-save">
            <p className="subtext">
              Revision {metadata?.number} · {metadata?.state}. Reviewed context
              can still change in later discussion.
            </p>
            <label className="context-check">
              <input type="checkbox" {...form.register("leaveRemaining")} />{" "}
              Leave all remaining proposals unresolved
            </label>
            <p className="subtext" role="status" aria-live="polite">
              {dirty
                ? `${edits.length} unsaved statement changes${leaveRemaining ? ", remaining proposals will be unresolved" : ""}.`
                : "All displayed changes are saved."}
            </p>
            {save.error && <ErrorMessage error={save.error} />}
            <div className="claim-actions">
              <Button
                variant="outline"
                disabled={save.isPending || !dirty}
                onClick={() => save.mutate("draft")}
              >
                Save draft
              </Button>
              <Button
                disabled={
                  save.isPending || (!dirty && metadata?.state === "reviewed")
                }
                onClick={() => save.mutate("reviewed")}
              >
                {save.isPending ? "Saving revision…" : "Save reviewed revision"}
              </Button>
            </div>
          </div>
        )}
      </section>
      {inspection && (
        <ReviewInspector
          inspection={inspection}
          close={() => setInspection(null)}
        >
          {inspection.entries.map(card)}
        </ReviewInspector>
      )}
      {editor && read.data && (
        <ClaimEditor
          entry={editor.entry}
          section={editor.section}
          workspace={workspace}
          concepts={read.data.content.concepts ?? {}}
          save={stage}
          close={() => setEditor(null)}
        />
      )}
    </div>
  );
}
