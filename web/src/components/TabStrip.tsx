import { TABS, type Tab } from "../selection";

export const LABEL: Record<Tab, string> = { claims: "Claims", runs: "Results", files: "Files" };
export type Counts = Partial<Record<Tab, string>>;

/** The slim strip that stands in for the inspector while it is closed. */
export default function TabStrip({ counts, onPick }: { counts: Counts; onPick: (t: Tab) => void }) {
  return (
    <div className="tabstrip" role="tablist" aria-orientation="vertical" aria-label="Working memory">
      {TABS.map((t) => (
        <button key={t} type="button" className="tab" role="tab" aria-selected={false} onClick={() => onPick(t)}>
          <span>{LABEL[t]}</span>
          {counts[t] && <span className="badge">{counts[t]}</span>}
        </button>
      ))}
    </div>
  );
}
