import { useEffect } from "react";
import type { SplitterProps } from "../hooks/usePaneResize";
import { TABS, type Selection } from "../selection";
import type { SessionView } from "../types";
import ClaimsTable from "./inspector/ClaimsTable";
import FilesView from "./inspector/FilesView";
import RunsView from "./inspector/RunsView";
import StatusMatrix from "./inspector/StatusMatrix";
import Splitter from "./Splitter";
import { LABEL, type Counts } from "./TabStrip";

/** The wide pane on the right: one tab at a time, with room for real tables. */
export default function Inspector({
  view,
  sel,
  counts,
  width,
  splitterProps,
  onSelect,
  onClose,
}: {
  view: SessionView;
  sel: NonNullable<Selection>;
  counts: Counts;
  width: number | null;
  splitterProps: SplitterProps;
  onSelect: (s: Selection) => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented || document.querySelector("dialog[open]")) return;
      onClose();
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [onClose]);

  return (
    <aside className="inspector" aria-label={`Working memory: ${LABEL[sel.tab]}`}>
      <Splitter width={width} {...splitterProps} />
      <div className="insp-head">
        <div className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={sel.tab === t}
              className={`tab${sel.tab === t ? " on" : ""}`}
              onClick={() => onSelect({ tab: t })}
            >
              <span>{LABEL[t]}</span>
              {counts[t] && <span className="badge">{counts[t]}</span>}
            </button>
          ))}
        </div>
        <button type="button" className="btn quiet sm" onClick={onClose}>
          Close
        </button>
      </div>
      <div className="insp-body">
        {sel.tab === "claims" && (
          <>
            <StatusMatrix status={view.status} claims={view.claims} />
            <ClaimsTable claims={view.claims} />
          </>
        )}
        {sel.tab === "runs" && <RunsView view={view} run={sel.run} onSelect={onSelect} />}
        {sel.tab === "files" && <FilesView view={view} run={sel.run} file={sel.file} onSelect={onSelect} />}
      </div>
    </aside>
  );
}
