import { Link } from "@tanstack/react-router";
import {
  BookOpen,
  Hash,
  ScanLine,
  Wrench,
  CircleHelp,
  Flag,
  Columns3,
  CircleDot,
  Plus,
  ArrowUpRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ContextTile, ReviewCounts } from "./overview";
import type { Inspection } from "./overview";
import { columnGroups, entries, label } from "./types";
import type { Entry, Read, Workspace } from "./types";
const icons = [BookOpen, Hash, ScanLine, Wrench, CircleHelp, Flag];

export function ColumnReview({
  columnId,
  read,
  workspace,
  inspect,
  preview,
  navigation,
  addRelationship,
}: {
  columnId: string;
  read: Read;
  workspace: Workspace;
  inspect: (selection: Inspection) => void;
  preview: (entry: Entry) => Entry;
  navigation: { datasetId: string; versionId: string; revision: string };
  addRelationship?: () => void;
}) {
  const section = `columns.${columnId}`;
  const current = (scope: string) =>
    entries(read.content, scope, workspace).filter(
      (e) => e.claim.status !== "withdrawn",
    );
  const own = current(section);
  const study = current("dataset").filter(
    (e) =>
      "column_ids" in e.claim &&
      (!e.claim.column_ids?.length || e.claim.column_ids.includes(columnId)),
  );
  const relationships = current("relationships").filter(
    (e) =>
      "participants" in e.claim &&
      e.claim.participants.some((p) => p.id === columnId),
  );
  const participants = Object.keys(read.related_to?.[columnId] ?? {});
  const definitions = participants.flatMap((id) => {
    const concept = read.content.concepts?.[id];
    if (concept) return [{ key: `concepts.${id}`, claim: concept }];
    return current(`columns.${id}`).filter((e) =>
      ["definition", "concept", "entity"].includes(e.key.split(".").at(-2)!),
    );
  });
  const name = (id: string) =>
    workspace.columns.find((c) => c.column_id === id)?.name ??
    read.content.concepts?.[id]?.value ??
    id;
  return (
    <>
      <div className="column-summary">
        <span className="eyebrow">COLUMN CONTEXT</span>
        <ReviewCounts entries={own.map(preview)} />
      </div>
      <div className="column-board">
        <div className="column-fields">
          <div className="context-tile-grid">
            {columnGroups.map((group, i) => {
              const claims = own.filter((e) =>
                (group.fields as readonly string[]).includes(
                  e.key.split(".").at(-2)!,
                ),
              );
              const shown = claims.map(preview);
              const Icon = icons[i]!;
              const filled = group.fields.filter((f) =>
                claims.some((e) => e.key.split(".").at(-2) === f),
              );
              return (
                <section key={group.title} aria-label={group.title}>
                  <ContextTile
                    title={group.title}
                    icon={<Icon size={17} aria-hidden="true" />}
                    entries={shown}
                    onOpen={() =>
                      inspect({
                        title: group.title,
                        entries: claims,
                        emptyFields: group.fields.filter(
                          (f) => !filled.includes(f),
                        ),
                      })
                    }
                  >
                    {shown.length ? (
                      shown.slice(0, 2).map((e) => (
                        <span className="tile-fact" key={e.key}>
                          <span className="tile-field-label">
                            {label(e.key.split(".").at(-2)!)}
                            {shown.filter(
                              (other) =>
                                other.key.split(".").at(-2) ===
                                e.key.split(".").at(-2),
                            ).length > 1 && (
                              <span className="field-accounts">
                                Multiple accounts
                              </span>
                            )}
                          </span>
                          <span className="tile-excerpt">{e.claim.value}</span>
                        </span>
                      ))
                    ) : (
                      <span className="tile-unknown">Not supplied</span>
                    )}
                    <span
                      className="field-coverage"
                      role="img"
                      aria-label={`${filled.length} of ${group.fields.length} fields supplied`}
                    >
                      {group.fields.map((f) => (
                        <i
                          key={f}
                          className={filled.includes(f) ? "filled" : ""}
                          title={label(f)}
                        />
                      ))}
                    </span>
                  </ContextTile>
                </section>
              );
            })}
          </div>
          <section
            className="study-context-strip"
            aria-label="Applicable study context"
          >
            <BookOpen size={17} aria-hidden="true" />
            <div>
              <h3>Study context</h3>
              <p>{study.length} applicable claims from the study account</p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() =>
                inspect({ title: "Applicable study context", entries: study })
              }
            >
              Inspect <ArrowUpRight size={13} aria-hidden="true" />
            </Button>
          </section>
        </div>
        <section className="related-panel" aria-label="Related to">
          <div className="related-heading">
            <h3>Related to</h3>
            <span className="tag">{relationships.length} accounts</span>
          </div>
          <div className="participant-map">
            <div className="participant-center">
              <Columns3 size={16} aria-hidden="true" />
              {name(columnId)}
            </div>
            <div className="participant-connector" aria-hidden="true" />
            <div className="participant-nodes">
              {participants.map((id) => (
                <Link
                  key={id}
                  to="/datasets/$datasetId/context/$versionId"
                  params={navigation}
                  search={{
                    revision: navigation.revision,
                    section: read.content.columns?.[id]
                      ? `columns.${id}`
                      : "concepts",
                  }}
                  className="participant-node"
                >
                  <CircleDot size={11} aria-hidden="true" />
                  <span>{name(id)}</span>
                </Link>
              ))}
            </div>
            <p>Participants in supplied accounts</p>
          </div>
          <div className="relationship-summaries">
            {relationships.map((entry, i) => {
              const e = preview(entry);
              return (
                <button
                  type="button"
                  key={e.key}
                  className="relationship-summary"
                  onClick={() =>
                    inspect({ title: "Relationship details", entries: [entry] })
                  }
                  aria-label={`Inspect relationship ${i + 1}`}
                >
                  <span className="relationship-summary-heading">
                    <span className={`review-count ${e.claim.status}`}>
                      <i aria-hidden="true" />
                      {e.claim.status === "proposed"
                        ? "To review"
                        : label(e.claim.status ?? "proposed")}
                    </span>
                    <ArrowUpRight size={13} aria-hidden="true" />
                  </span>
                  <span className="tile-excerpt">{e.claim.value}</span>
                  {"participants" in e.claim && (
                    <span className="relationship-members">
                      {e.claim.participants.length} participants ·{" "}
                      {e.claim.citations?.length ?? 0} citations
                    </span>
                  )}
                </button>
              );
            })}
            {!relationships.length && (
              <p className="subtext p-4">No relationship claims supplied.</p>
            )}
          </div>
          <div className="related-footer">
            {definitions.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  inspect({
                    title: "Participant definitions",
                    entries: definitions,
                  })
                }
              >
                Participant definitions
              </Button>
            )}
            {addRelationship && (
              <Button variant="ghost" size="sm" onClick={addRelationship}>
                <Plus size={13} aria-hidden="true" />
                Add relationship
              </Button>
            )}
          </div>
        </section>
      </div>
      <details className="context-empty-fields context-retrieval-keys">
        <summary>Inspect retrieval keys</summary>
        <ul>
          {read.included_keys.map((key) => (
            <li key={key}>
              <code>{key}</code>
            </li>
          ))}
        </ul>
      </details>
    </>
  );
}
