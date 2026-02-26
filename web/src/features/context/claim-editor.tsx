import { useFieldArray, useForm } from "react-hook-form";
import { useBlocker } from "@tanstack/react-router";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import type { Concept, Edit, Entry, Relation, Workspace } from "./types";
import { label } from "./types";

const schema = z.object({
  field: z.string(),
  value: z.string().trim().min(1, "Enter the statement.").max(10000),
  basis: z.enum(["source_statement", "interpretation"]),
  qualifications: z.string().max(10000),
  reason: z.string().max(10000),
  kind: z.string(),
  timing: z.string(),
  conditions: z.string(),
  directness: z.enum(["direct", "indirect", "unspecified"]),
  absence_scope: z.enum(["direct_effect", "any_pathway", "unspecified"]),
  column_ids: z.array(z.string()),
  participants: z.array(
    z.object({
      id: z.string().min(1),
      kind: z.enum(["column", "concept", "event"]),
      role: z.enum(["participant", "from", "to", "intermediate", "condition"]),
    }),
  ),
  keepCitations: z.boolean(),
});
type Values = z.infer<typeof schema>;

export function ClaimEditor({
  entry,
  section,
  workspace,
  concepts,
  save,
  close,
}: {
  entry?: Entry;
  section: string;
  workspace: Workspace;
  concepts: Record<string, Concept>;
  save: (edit: Edit) => void;
  close: () => void;
}) {
  const relationship = section === "relationships";
  const concept = section === "concepts";
  const relation = entry?.claim as Relation | undefined;
  const fields =
    section === "dataset" ? workspace.dataset_fields : workspace.column_fields;
  const form = useForm<Values>({
    resolver: zodResolver(schema),
    defaultValues: {
      field: entry?.key.split(".").at(-2) ?? fields[0],
      value: entry?.claim.value ?? "",
      basis: entry?.claim.basis ?? "source_statement",
      qualifications: entry?.claim.qualifications ?? "",
      reason: "",
      kind: relation?.kind ?? (concept ? "concept" : ""),
      timing: relation?.timing ?? "",
      conditions: relation?.conditions ?? "",
      directness: relation?.directness ?? "unspecified",
      absence_scope: relation?.absence_scope ?? "unspecified",
      column_ids:
        entry && "column_ids" in entry.claim
          ? (entry.claim.column_ids ?? [])
          : [],
      participants: relation?.participants ?? [],
      keepCitations: false,
    },
  });
  const participants = useFieldArray({
    control: form.control,
    name: "participants",
  });
  useBlocker({
    shouldBlockFn: () =>
      form.formState.isDirty &&
      !window.confirm("Discard these unsaved statement changes?"),
    enableBeforeUnload: form.formState.isDirty,
  });
  const dismiss = () => {
    if (
      !form.formState.isDirty ||
      window.confirm("Discard these unsaved statement changes?")
    )
      close();
  };
  function submit(values: Values) {
    if (entry && !values.reason.trim()) {
      form.setError("reason", { message: "Explain the correction." });
      return;
    }
    if (
      relationship &&
      new Set(values.participants.map((p) => p.id)).size < 2
    ) {
      form.setError("participants", {
        message: "Select at least two distinct participants.",
      });
      return;
    }
    const { field, reason, keepCitations, ...statement } = values;
    save({
      action: entry ? "replace" : "add",
      key:
        entry?.key ??
        (relationship || concept ? section : `${section}.${field}`),
      reason,
      statement: {
        ...statement,
        kind: statement.kind || null,
        citations: keepCitations ? (entry?.claim.citations ?? []) : [],
      },
    });
    close();
  }
  return (
    <Dialog
      open
      onOpenChange={(v) => {
        if (!v) dismiss();
      }}
    >
      <DialogContent
        className="claim-dialog"
        showCloseButton={false}
        onEscapeKeyDown={(e) => {
          e.preventDefault();
          dismiss();
        }}
        onInteractOutside={(e) => e.preventDefault()}
      >
        <DialogTitle>
          {entry ? "Correct statement" : "Add statement"}
        </DialogTitle>
        <DialogDescription>
          Your wording is recorded as a user contribution. Saving the revision
          preserves the earlier account.
        </DialogDescription>
        <form onSubmit={form.handleSubmit(submit)} noValidate>
          {!entry && !relationship && !concept && (
            <div className="form-field">
              <Label htmlFor="claim-field">Context field</Label>
              <select
                id="claim-field"
                className="context-input"
                {...form.register("field")}
              >
                {fields.map((f) => (
                  <option key={f} value={f}>
                    {label(f)} ({f})
                  </option>
                ))}
              </select>
            </div>
          )}
          {(relationship || concept) && (
            <div className="form-field">
              <Label htmlFor="claim-kind">
                {relationship ? "Relationship kind (optional)" : "Concept kind"}
              </Label>
              <select
                id="claim-kind"
                className="context-input"
                {...form.register("kind")}
              >
                {relationship && <option value="">Not classified</option>}
                {(concept
                  ? ["concept", "event"]
                  : workspace.relationship_kinds
                ).map((k) => (
                  <option key={k} value={k}>
                    {label(k)}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="form-field">
            <Label htmlFor="claim-value">Statement</Label>
            <textarea
              id="claim-value"
              className="context-input"
              rows={4}
              {...form.register("value")}
              aria-invalid={!!form.formState.errors.value}
            />
            <span className="field-error">
              {form.formState.errors.value?.message}
            </span>
          </div>
          <div className="form-field">
            <Label htmlFor="claim-basis">Basis</Label>
            <select
              id="claim-basis"
              className="context-input"
              {...form.register("basis")}
            >
              <option value="source_statement">Supplied statement</option>
              <option value="interpretation">Interpretation</option>
            </select>
          </div>
          <div className="form-field">
            <Label htmlFor="claim-qualifications">Qualifications</Label>
            <textarea
              id="claim-qualifications"
              className="context-input"
              rows={2}
              {...form.register("qualifications")}
            />
          </div>
          {section === "dataset" && (
            <fieldset className="form-field">
              <legend>Applies to columns</legend>
              <p className="subtext">Leave unselected for the whole dataset.</p>
              <div className="column-options">
                {workspace.columns.map((c) => (
                  <label key={c.column_id}>
                    <input
                      type="checkbox"
                      value={c.column_id}
                      {...form.register("column_ids")}
                    />{" "}
                    {c.name}
                  </label>
                ))}
              </div>
            </fieldset>
          )}
          {relationship && (
            <>
              <fieldset className="form-field">
                <legend>Participants</legend>
                {participants.fields.map((p, index) => (
                  <div key={p.id} className="participant-row">
                    <select
                      aria-label={`Participant ${index + 1}`}
                      className="context-input"
                      value={form.watch(`participants.${index}.id`)}
                      onChange={(e) => {
                        const id = e.target.value;
                        const kind = workspace.columns.some(
                          (c) => c.column_id === id,
                        )
                          ? "column"
                          : (concepts[id]?.kind ?? "concept");
                        form.setValue(`participants.${index}.id`, id, {
                          shouldDirty: true,
                        });
                        form.setValue(`participants.${index}.kind`, kind);
                      }}
                    >
                      <option value="">Select participant</option>
                      {workspace.columns.map((c) => (
                        <option key={c.column_id} value={c.column_id}>
                          {c.name}
                        </option>
                      ))}
                      {Object.entries(concepts).map(([id, c]) => (
                        <option key={id} value={id}>
                          {c.value} ({c.kind})
                        </option>
                      ))}
                    </select>
                    <select
                      aria-label={`Participant ${index + 1} role`}
                      className="context-input"
                      {...form.register(`participants.${index}.role`)}
                    >
                      {[
                        "participant",
                        "from",
                        "to",
                        "intermediate",
                        "condition",
                      ].map((r) => (
                        <option key={r}>{r}</option>
                      ))}
                    </select>
                    <Button
                      type="button"
                      variant="ghost"
                      onClick={() => participants.remove(index)}
                      aria-label={`Remove participant ${index + 1}`}
                    >
                      Remove
                    </Button>
                  </div>
                ))}
                <span className="field-error">
                  {form.formState.errors.participants?.message}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    participants.append({
                      id: "",
                      kind: "column",
                      role: "participant",
                    })
                  }
                >
                  Add participant
                </Button>
              </fieldset>
              <div className="form-field">
                <Label htmlFor="relation-timing">Timing</Label>
                <textarea
                  id="relation-timing"
                  className="context-input"
                  {...form.register("timing")}
                />
              </div>
              <div className="form-field">
                <Label htmlFor="relation-conditions">Conditions</Label>
                <textarea
                  id="relation-conditions"
                  className="context-input"
                  {...form.register("conditions")}
                />
              </div>
              <div className="form-field">
                <Label htmlFor="relation-directness">Directness</Label>
                <select
                  id="relation-directness"
                  className="context-input"
                  {...form.register("directness")}
                >
                  <option value="unspecified">Unspecified</option>
                  <option value="direct">Direct</option>
                  <option value="indirect">Indirect</option>
                </select>
              </div>
              <div className="form-field">
                <Label htmlFor="relation-absence">Absence scope</Label>
                <select
                  id="relation-absence"
                  className="context-input"
                  {...form.register("absence_scope")}
                >
                  <option value="unspecified">Unspecified</option>
                  <option value="direct_effect">Direct effect</option>
                  <option value="any_pathway">Any pathway</option>
                </select>
              </div>
            </>
          )}
          {entry?.claim.citations?.length ? (
            <label className="context-check">
              <input type="checkbox" {...form.register("keepCitations")} />{" "}
              These original passages also support my correction.
            </label>
          ) : null}
          <div className="form-field">
            <Label htmlFor="claim-reason">
              {entry ? "Reason for correction" : "Additional note (optional)"}
            </Label>
            <textarea
              id="claim-reason"
              className="context-input"
              rows={2}
              {...form.register("reason")}
              aria-invalid={!!form.formState.errors.reason}
            />
            <span className="field-error">
              {form.formState.errors.reason?.message}
            </span>
          </div>
          <div className="modal-actions">
            <Button type="button" variant="outline" onClick={dismiss}>
              Cancel
            </Button>
            <Button type="submit">Stage change</Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
