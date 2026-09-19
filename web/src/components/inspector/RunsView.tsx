import type { Selection } from "../../selection";
import type { SessionView } from "../../types";
import RunDetail from "./RunDetail";

export default function RunsView({ view, run, onSelect }: { view: SessionView; run: number | undefined; onSelect: (s: Selection) => void }) {
  if (!view.runs.length) return <p className="empty-note">No run yet. Say run once the claims are settled.</p>;
  const r = view.runs.find((x) => x.index === run) ?? view.runs[view.runs.length - 1];
  const i = view.runs.indexOf(r);
  return (
    <>
      {view.runs.length > 1 && (
        <div className="picker" role="tablist" aria-label="Runs">
          {view.runs.map((x) => (
            <button key={x.index} type="button" role="tab" aria-selected={x.index === r.index} className={`chip${x.index === r.index ? " on" : ""}`} onClick={() => onSelect({ tab: "runs", run: x.index })}>
              {Object.keys(x.what_if ?? {}).length ? "What if " : "Design "}
              {x.index}
              {x.family ? ` · ${x.family}` : ""}
            </button>
          ))}
        </div>
      )}
      <RunDetail key={r.index} r={r} prev={i > 0 ? view.runs[i - 1] : null} onSelect={onSelect} />
    </>
  );
}
