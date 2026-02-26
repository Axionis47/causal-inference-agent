import { useRef } from "react";
import type { ReactNode } from "react";
import { ArrowUpRight, FileText } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { label } from "./types";
import type { Entry } from "./types";

export type Inspection = {
  title: string;
  entries: Entry[];
  emptyFields?: string[];
};

export function ReviewCounts({ entries }: { entries: Entry[] }) {
  const counts = ["proposed", "reviewed", "unresolved"].map((status) => ({
    status,
    count: entries.filter((e) => e.claim.status === status).length,
  }));
  return (
    <div className="review-counts" aria-label="Statement review status">
      {counts
        .filter((c) => c.count)
        .map((c) => (
          <span key={c.status} className={`review-count ${c.status}`}>
            <i aria-hidden="true" />
            {c.count} {c.status === "proposed" ? "to review" : c.status}
          </span>
        ))}
      {!entries.length && (
        <span className="review-count">No supplied claims</span>
      )}
    </div>
  );
}

export function ContextTile({
  title,
  icon,
  entries,
  onOpen,
  children,
}: {
  title: string;
  icon?: ReactNode;
  entries: Entry[];
  onOpen: () => void;
  children?: ReactNode;
}) {
  const sources = new Set(
    entries.flatMap((e) => e.claim.citations?.map((c) => c.source_id) ?? []),
  );
  return (
    <button
      type="button"
      className={`context-tile ${entries.length ? "has-content" : "is-empty"}`}
      onClick={onOpen}
      aria-label={`Inspect ${title}`}
    >
      <span className="tile-heading">
        <span className="tile-icon">
          {icon ?? <FileText size={16} aria-hidden="true" />}
        </span>
        <span>{title}</span>
        <ArrowUpRight className="tile-open" size={15} aria-hidden="true" />
      </span>
      <span className="tile-body">
        {children ??
          (entries.length ? (
            entries.slice(0, 2).map((e) => (
              <span key={e.key} className="tile-excerpt">
                {e.claim.value}
              </span>
            ))
          ) : (
            <span className="tile-unknown">Not supplied</span>
          ))}
      </span>
      <span className="tile-footer">
        <span>
          {entries.length} {entries.length === 1 ? "claim" : "claims"}
        </span>
        <span>
          {sources.size
            ? `${sources.size} ${sources.size === 1 ? "source" : "sources"}`
            : "Open to inspect"}
        </span>
      </span>
    </button>
  );
}

export function ReviewInspector({
  inspection,
  close,
  children,
}: {
  inspection: Inspection;
  close: () => void;
  children: ReactNode;
}) {
  const opener = useRef(document.activeElement as HTMLElement | null);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) close();
      }}
    >
      <DialogContent
        className="context-inspector"
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          opener.current?.focus();
        }}
      >
        <DialogTitle>{inspection.title}</DialogTitle>
        <DialogDescription>
          Inspect the account, check its source, and stage review changes.
        </DialogDescription>
        <div className="inspector-content">
          {children}
          {!inspection.entries.length && (
            <p className="subtext">
              No statements supplied. This can remain unknown.
            </p>
          )}
          {!!inspection.emptyFields?.length && (
            <details className="context-empty-fields">
              <summary>
                {inspection.emptyFields.length} empty retrieval fields
              </summary>
              <ul>
                {inspection.emptyFields.map((f) => (
                  <li key={f}>
                    {label(f)} <code>{f}</code>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
