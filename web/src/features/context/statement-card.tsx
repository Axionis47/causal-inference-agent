import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { label } from "./types";
import { ContextReferences } from "./analysis-brief";
import type { Citation, Edit, Entry, Read, Workspace } from "./types";

export function StatementCard({
  entry,
  pending,
  workspace,
  read,
  navigation,
  onCitation,
  onNavigate,
  stage,
  correct,
  undo,
  disabled,
}: {
  entry: Entry;
  pending?: Edit;
  workspace: Workspace;
  read: Read;
  navigation: { datasetId: string; versionId: string; revision: string };
  onCitation: (cite: Citation) => void;
  onNavigate: () => void;
  stage?: (edit: Edit) => void;
  correct: () => void;
  undo: () => void;
  disabled: boolean;
}) {
  const claim = entry.claim;
  const wording = pending?.statement ?? claim;
  const relationship =
    "participants" in claim ? { ...claim, ...pending?.statement } : null;
  return (
    <article key={entry.key} className="claim-card" data-testid="claim-card">
      <div className="claim-heading">
        <span className="tag">
          {pending
            ? `UNSAVED ${pending.action.toUpperCase()}`
            : claim.status?.toUpperCase()}
        </span>
        <span className="eyebrow">
          {label(
            "kind" in claim
              ? ((pending?.statement ? pending.statement.kind : claim.kind) ??
                  "Relationship")
              : (entry.key.split(".").at(-2) ?? ""),
          )}
        </span>
      </div>
      <p className="claim-value">{wording.value}</p>
      <p className="context-key">{entry.key}</p>
      {wording.qualifications && (
        <p className="subtext">Qualifications: {wording.qualifications}</p>
      )}
      {!!wording.conflicts_with?.length && (
        <div className="mt-3">
          <p className="eyebrow">Declared conflicts</p>
          <ContextReferences
            references={wording.conflicts_with}
            location={navigation}
            onNavigate={onNavigate}
          />
        </div>
      )}
      {relationship && (
        <>
          <ul className="relationship-participants">
            {relationship.participants.map((p, i) => (
              <li key={`${p.id}-${i}`}>
                <span className="text-muted-foreground">
                  {p.role ?? "participant"}:{" "}
                </span>
                <Link
                  to="/datasets/$datasetId/context/$versionId"
                  params={navigation}
                  search={{
                    revision: navigation.revision,
                    section:
                      p.kind === "column" ? `columns.${p.id}` : "concepts",
                  }}
                  className="context-link"
                  onClick={onNavigate}
                >
                  {workspace.columns.find((c) => c.column_id === p.id)?.name ??
                    read.content.concepts?.[p.id]?.value ??
                    p.id}
                </Link>
              </li>
            ))}
          </ul>
          {relationship.timing && (
            <p className="subtext">Timing: {relationship.timing}</p>
          )}
          {relationship.conditions && (
            <p className="subtext">Conditions: {relationship.conditions}</p>
          )}
          <p className="subtext">
            Directness: {label(relationship.directness ?? "unspecified")} ·
            Absence scope: {label(relationship.absence_scope ?? "unspecified")}
          </p>
        </>
      )}
      <div className="claim-evidence">
        <span>
          {wording.basis === "interpretation"
            ? "Interpretation"
            : "Supplied statement"}{" "}
          ·{" "}
          {pending?.statement
            ? "User correction (unsaved)"
            : claim.source.kind === "extraction"
              ? "Model prepared"
              : "User supplied"}
        </span>
        {wording.citations?.map((cite, i) => (
          <button
            key={i}
            type="button"
            onClick={() => onCitation(cite)}
            className="context-link"
          >
            {read?.sources[cite.source_id]?.title ?? "Source"}: line{" "}
            {cite.source_line}
          </button>
        ))}
      </div>
      {claim.reason && <p className="subtext">Review note: {claim.reason}</p>}
      {claim.status !== "withdrawn" && stage && (
        <div className="claim-actions">
          <Button
            variant="outline"
            size="sm"
            disabled={disabled || (!pending && claim.status === "reviewed")}
            onClick={() =>
              stage?.({ action: "review", key: entry.key, reason: "" })
            }
          >
            Mark reviewed
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={disabled || (!pending && claim.status === "unresolved")}
            onClick={() =>
              stage?.({ action: "unresolved", key: entry.key, reason: "" })
            }
          >
            Leave unresolved
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={disabled}
            onClick={correct}
          >
            Correct
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={disabled}
            onClick={() => {
              const reason = window.prompt(
                "Why should this statement be withdrawn?",
              );
              if (reason?.trim())
                stage?.({ action: "withdraw", key: entry.key, reason });
            }}
          >
            Withdraw
          </Button>
          {pending && (
            <Button variant="ghost" size="sm" onClick={undo}>
              Undo change
            </Button>
          )}
        </div>
      )}
    </article>
  );
}
