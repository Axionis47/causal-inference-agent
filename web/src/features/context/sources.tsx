import { useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { useMutation } from "@tanstack/react-query";
import { useBlocker } from "@tanstack/react-router";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { api, client, errorMessage, fileSize } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { ErrorMessage, Loading } from "@/components/feedback";
import type { Citation, Source, Workspace } from "./types";

const uploadSchema = z
  .object({
    title: z.string().trim().min(1, "Give this source a title.").max(200),
    kind: z.enum(["file", "note"]),
    note: z.string(),
    file: z.instanceof(File).optional(),
  })
  .superRefine((data, ctx) => {
    if (data.kind === "note" && !data.note.trim())
      ctx.addIssue({
        code: "custom",
        path: ["note"],
        message: "Enter your source notes.",
      });
    if (
      data.kind === "file" &&
      (!data.file || !/\.(txt|md)$/i.test(data.file.name))
    )
      ctx.addIssue({
        code: "custom",
        path: ["file"],
        message: "Choose a .txt or .md file.",
      });
    const size =
      data.kind === "note"
        ? new Blob([data.note]).size
        : (data.file?.size ?? 0);
    if (size > 1024 * 1024)
      ctx.addIssue({
        code: "custom",
        path: [data.kind === "note" ? "note" : "file"],
        message: "The source exceeds 1 MiB.",
      });
  });
type UploadValues = z.infer<typeof uploadSchema>;

function SourceForm({
  versionId,
  saved,
  close,
}: {
  versionId: string;
  saved: () => Promise<void>;
  close: () => void;
}) {
  const form = useForm<UploadValues>({
    resolver: zodResolver(uploadSchema),
    defaultValues: { title: "", kind: "file", note: "" },
  });
  const submission = useRef<{ signature: string; id: string } | null>(null);
  const kind = form.watch("kind");
  const upload = useMutation({
    mutationFn: async (values: UploadValues) => {
      const file =
        values.kind === "note"
          ? new File([values.note], "notes.txt", { type: "text/plain" })
          : values.file!;
      const signature = JSON.stringify([
        values.title,
        values.kind,
        file.name,
        Array.from(
          new Uint8Array(
            await crypto.subtle.digest("SHA-256", await file.arrayBuffer()),
          ),
        )
          .map((byte) => byte.toString(16).padStart(2, "0"))
          .join(""),
      ]);
      if (submission.current?.signature !== signature)
        submission.current = { signature, id: crypto.randomUUID() };
      const body = {
        title: values.title,
        kind: values.kind,
        submission_id: submission.current.id,
        file: file.name,
      };
      const result = await client.POST(
        "/api/versions/{version_id}/context/sources",
        {
          params: { path: { version_id: versionId } },
          body,
          bodySerializer: () => {
            const data = new FormData();
            data.set("title", body.title);
            data.set("kind", body.kind);
            data.set("submission_id", body.submission_id);
            data.set("file", file);
            return data;
          },
        },
      );
      if (result.error) throw new Error(errorMessage(result.error));
    },
    onSuccess: async () => {
      form.reset();
      await saved();
      close();
    },
  });
  const dirty = form.formState.isDirty;
  useBlocker({
    shouldBlockFn: () =>
      dirty && !window.confirm("Discard this unsaved source?"),
    enableBeforeUnload: dirty,
  });
  const dismiss = () => {
    if (
      !upload.isPending &&
      (!dirty || window.confirm("Discard this unsaved source?"))
    )
      close();
  };
  return (
    <DialogContent
      showCloseButton={false}
      onEscapeKeyDown={(e) => {
        e.preventDefault();
        dismiss();
      }}
      onInteractOutside={(e) => e.preventDefault()}
    >
      <DialogTitle>Add source material</DialogTitle>
      <DialogDescription>
        Attach text or Markdown to this exact dataset version. Up to 1 MiB per
        source.
      </DialogDescription>
      <form onSubmit={form.handleSubmit((v) => upload.mutate(v))} noValidate>
        <fieldset disabled={upload.isPending}>
          <div className="form-field">
            <Label htmlFor="source-title">Source title</Label>
            <Input
              id="source-title"
              autoComplete="off"
              {...form.register("title")}
              aria-invalid={!!form.formState.errors.title}
            />
            <span className="field-error">
              {form.formState.errors.title?.message}
            </span>
          </div>
          <div className="form-field">
            <Label htmlFor="source-kind">Source format</Label>
            <select
              id="source-kind"
              className="context-input"
              {...form.register("kind")}
            >
              <option value="file">Upload text or Markdown</option>
              <option value="note">Paste notes</option>
            </select>
          </div>
          {kind === "note" ? (
            <div className="form-field">
              <Label htmlFor="source-note">Source notes</Label>
              <textarea
                id="source-note"
                rows={8}
                className="context-input"
                {...form.register("note")}
                aria-invalid={!!form.formState.errors.note}
              />
              <span className="field-error">
                {form.formState.errors.note?.message}
              </span>
            </div>
          ) : (
            <div className="form-field">
              <Label htmlFor="source-file">Text or Markdown file</Label>
              <input
                id="source-file"
                type="file"
                accept=".txt,.md,text/plain,text/markdown"
                className="file-input"
                onChange={(e) =>
                  form.setValue("file", e.target.files?.[0], {
                    shouldDirty: true,
                    shouldValidate: true,
                  })
                }
              />
              <span className="field-error">
                {form.formState.errors.file?.message}
              </span>
            </div>
          )}
          {upload.error && <ErrorMessage error={upload.error} />}
          <div className="modal-actions">
            <Button
              type="button"
              variant="outline"
              onClick={dismiss}
              disabled={upload.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={upload.isPending}>
              {upload.isPending ? "Saving source…" : "Save source"}
            </Button>
          </div>
        </fieldset>
      </form>
    </DialogContent>
  );
}

export function Sources({
  versionId,
  workspace,
  included,
  onInspect,
  refresh,
}: {
  versionId: string;
  workspace: Workspace;
  included: string[];
  onInspect: (source: Source) => void;
  refresh: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  return (
    <section className="panel mb-6" aria-labelledby="sources-title">
      <div className="panel-heading">
        <h2 id="sources-title" className="panel-title">
          01 / Source material
        </h2>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setOpen(true)}
          disabled={workspace.sources.length >= 10}
        >
          Add source
        </Button>
      </div>
      {workspace.sources.length ? (
        <ul className="source-list">
          {workspace.sources.map((source) => (
            <li key={source.id}>
              <button
                onClick={() => onInspect(source)}
                className="source-title"
              >
                {source.title}
                <span className="filename">
                  {source.filename} · {fileSize(source.size_bytes)}
                </span>
              </button>
              <span className="tag">
                {included.includes(source.id)
                  ? "IN CURRENT REVISION"
                  : "NOT IN CURRENT REVISION"}
              </span>
              <a
                href={`/api/versions/${versionId}/context/sources/${source.id}/file`}
                className="context-link"
              >
                Original
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p className="p-5 subtext">
          Add a study account, codebook, or notes. Original wording and source
          passages stay available.
        </p>
      )}
      <Dialog open={open} onOpenChange={setOpen}>
        {open && (
          <SourceForm
            versionId={versionId}
            saved={refresh}
            close={() => setOpen(false)}
          />
        )}
      </Dialog>
    </section>
  );
}

export function SourceViewer({
  versionId,
  sourceId,
  citation,
  close,
}: {
  versionId: string;
  sourceId: string;
  citation?: Citation;
  close: () => void;
}) {
  const result = api.useQuery(
    "get",
    "/api/versions/{version_id}/context/sources/{source_id}",
    { params: { path: { version_id: versionId, source_id: sourceId } } },
  );
  const characters = Array.from(result.data?.text ?? "");
  return (
    <Dialog
      open
      onOpenChange={(v) => {
        if (!v) close();
      }}
    >
      <DialogContent className="source-dialog">
        <DialogTitle>
          {result.data?.source.title ?? "Source material"}
        </DialogTitle>
        <DialogDescription>
          {citation
            ? `Supporting passage, line ${citation.source_line}.`
            : "Original supplied text, with its wording and line breaks preserved."}
        </DialogDescription>
        {result.isPending ? (
          <Loading />
        ) : result.error ? (
          <ErrorMessage error={result.error} />
        ) : (
          <pre className="source-text" tabIndex={0}>
            {citation ? (
              <>
                {characters.slice(0, citation.start).join("")}
                <mark
                  ref={(node) => {
                    node?.scrollIntoView({ block: "center" });
                  }}
                >
                  {characters.slice(citation.start, citation.end).join("")}
                </mark>
                {characters.slice(citation.end).join("")}
              </>
            ) : (
              result.data?.text
            )}
          </pre>
        )}
      </DialogContent>
    </Dialog>
  );
}
