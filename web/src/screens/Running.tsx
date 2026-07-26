/** Screen 4: the live tape.
 *
 *  Rows arrive amber-tinted and cool to default within two seconds. That is the
 *  live/settled axis: motion lives in the transition and nowhere else.
 */
import type { TapeEvent } from "../api";
import { Dot, Pane, type Status } from "../ui";

const TONE: Record<TapeEvent["event"], Status> = {
  stage_started: "live",
  stage_done: "ok",
  waiting_for_you: "info",
  completed: "ok",
  failed: "failed",
};

export function Running({ tape, done }: { tape: TapeEvent[]; done: boolean }) {
  return (
    <div className="grid grid-cols-[1fr_320px] gap-4 h-full min-h-0">
      <Pane
        caption="tape"
        className="min-h-0"
        right={
          <span className="flex items-center gap-1.5">
            <Dot status={done ? "ok" : "live"} />
            <span className="text-2xs font-mono uppercase tracking-label text-ink-tertiary">
              {done ? "settled" : "live"}
            </span>
          </span>
        }
      >
        <div className="space-y-0 overflow-auto max-h-[calc(100vh-220px)] font-mono text-xs">
          {tape.map((e, i) => (
            <div
              key={i}
              className="flex items-baseline gap-3 py-1.5 border-b border-edge-subtle/50 last:border-0 animate-tape-arrival"
            >
              <span className="text-ink-tertiary tabular shrink-0 w-8 text-right">
                {String(i + 1).padStart(2, "0")}
              </span>
              <Dot status={TONE[e.event]} />
              <span className="text-ink-secondary uppercase tracking-label text-2xs w-28 shrink-0">
                {e.event.replace(/_/g, " ")}
              </span>
              <span className="text-ink flex-1">
                {e.stage ? <span className="text-indigo">{e.stage}</span> : null}
                {e.stage && (e.detail || e.reason) ? "  " : ""}
                {e.detail || e.reason || ""}
              </span>
            </div>
          ))}
          {!tape.length && (
            <p className="text-ink-tertiary py-6 text-center">waiting for the first event…</p>
          )}
        </div>
      </Pane>

      <Pane caption="stages">
        <div className="space-y-1.5">
          {["read", "menu", "reason", "gate", "estimate"].map((s) => {
            const started = tape.some((e) => e.stage === s);
            const finished = tape.some((e) => e.stage === s && e.event === "stage_done");
            return (
              <div key={s} className="flex items-center gap-2 py-1">
                <Dot status={finished ? "ok" : started ? "live" : "pending"} />
                <span className="text-xs font-mono text-ink-secondary">{s}</span>
              </div>
            );
          })}
        </div>
        <p className="text-2xs text-ink-tertiary leading-relaxed pt-3 mt-3 border-t border-edge-subtle">
          The tape replays from the start when you reconnect. Events live on the run's
          checkpoint, not in a queue that forgets.
        </p>
      </Pane>
    </div>
  );
}
