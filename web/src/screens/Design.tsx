/** Screen 3: the gate. Which designs this data supports, and which it doesn't.
 *
 *  The important screen. Every design is listed, available or not, each with
 *  the assumption it rests on and the reason it was ruled in or out. A design
 *  the data cannot reveal (an instrument, a mediator) is shown as a candidate
 *  awaiting a column, never silently dropped.
 *
 *  You pick. Nothing here guesses on your behalf.
 */
import { useState } from "react";
import type { Job } from "../api";
import { Button, Dot, Label, Pane } from "../ui";

/** What each lane needs, in call order. Prefilled from the reading where the
 *  engine already worked it out; everything stays editable. */
const PARAMS: Record<string, string[]> = {
  observational: ["outcome", "treatment", "covariates"],
  matching: ["outcome", "treatment", "covariates"],
  iv: ["outcome", "treatment", "instrument", "covariates"],
  did: ["outcome", "group", "period", "treated_group", "unit"],
  rdd: ["outcome", "running", "cutoff"],
  survival: ["treatment", "duration", "event", "covariates"],
  mediation: ["outcome", "treatment", "mediator", "covariates"],
  time_series: ["outcome", "time", "intervention"],
};

function prefill(lane: string, job: Job): Record<string, string> {
  const i = job.intake;
  // the engine proposes a full argument set per lane; use it where it has one
  const proposed = (job.suggestions ?? {})[lane] ?? {};
  const out: Record<string, string> = {};
  for (const p of PARAMS[lane] ?? []) {
    const given = (proposed as Record<string, unknown>)[p];
    if (given !== undefined && given !== null && given !== "") {
      out[p] = Array.isArray(given) ? given.join(", ") : String(given);
      continue;
    }
    out[p] =
      p === "outcome" ? i?.outcome ?? "" :
      p === "treatment" ? i?.treatment ?? "" :
      p === "group" ? i?.group ?? "" :
      p === "period" ? i?.period ?? "" :
      p === "time" ? i?.time_column ?? "" :
      p === "running" ? i?.running_variable ?? "" :
      p === "cutoff" ? (i?.cutoff != null ? String(i.cutoff) : "") :
      "";
  }
  return out;
}

export function Design({
  job,
  onChosen,
}: {
  job: Job;
  onChosen: (lane: string, kwargs: Record<string, unknown>) => void;
}) {
  const [lane, setLane] = useState("");
  const [args, setArgs] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);

  function select(next: string) {
    setLane(next);
    setArgs(prefill(next, job));
  }

  function go() {
    setBusy(true);
    const kwargs: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(args)) {
      if (!v.trim()) continue;
      if (k === "covariates") kwargs[k] = v.split(",").map((s) => s.trim()).filter(Boolean);
      else if (k === "cutoff") kwargs[k] = Number(v);
      else kwargs[k] = v.trim();
    }
    onChosen(lane, kwargs);
  }

  const menu = job.menu ?? [];
  const ready = lane && (PARAMS[lane] ?? []).every(
    (p) => p === "covariates" || p === "unit" || args[p]?.trim()
  );

  const rec = job.recommendation;

  return (
    <div className="grid grid-cols-[1fr_380px] gap-4 h-full min-h-0">
      <div className="min-h-0 flex flex-col gap-3">
      {rec?.lane && (
        <div className="border border-edge-subtle bg-canvas-raised px-3 py-2.5">
          <div className="flex items-center gap-2">
            <Dot status="info" />
            <Label>recommended</Label>
            <span className="text-xs font-mono text-amber">{rec.lane}</span>
            <span className="text-2xs font-mono text-ink-tertiary">
              step {rec.step} of the procedure · {rec.confidence} confidence
            </span>
            {lane !== rec.lane && (
              <button
                onClick={() => select(rec.lane)}
                className="ml-auto text-2xs font-mono uppercase tracking-label text-ink-secondary hover:text-ink border border-edge px-2 py-0.5"
              >
                use it
              </button>
            )}
          </div>
          <p className="text-xs text-ink-secondary mt-1.5 ml-3.5 leading-relaxed">
            {rec.reasoning}
          </p>
          {rec.missing && (
            <p className="text-2xs text-amber mt-1 ml-3.5 leading-relaxed">
              would be surer if you said: {rec.missing}
            </p>
          )}
          <p className="text-2xs text-ink-tertiary mt-1 ml-3.5">
            a recommendation, not a decision — override it below
          </p>
        </div>
      )}
      <Pane
        caption="designs"
        className="min-h-0"
        right={
          <span className="text-2xs font-mono text-ink-tertiary tabular">
            {menu.filter((m) => m.available).length} of {menu.length} available
          </span>
        }
      >
        <div className="space-y-1.5 overflow-auto max-h-[calc(100vh-220px)] pr-1">
          {menu.map((o) => {
            const chosen = lane === o.lane;
            return (
              <button
                key={o.lane}
                onClick={() => select(o.lane)}
                className={`w-full text-left px-3 py-2.5 border transition-colors ${
                  chosen
                    ? "border-amber-dim bg-canvas-overlay"
                    : o.available
                    ? "border-edge-subtle hover:border-edge"
                    : "border-edge-subtle/50 hover:border-edge-subtle"
                }`}
              >
                <div className="flex items-center gap-2">
                  <Dot status={chosen ? "live" : o.available ? "ok" : "pending"} />
                  <span
                    className={`text-xs font-mono ${
                      o.available ? "text-ink" : "text-ink-secondary"
                    }`}
                  >
                    {o.lane}
                  </span>
                  {!o.available && (
                    <span className="text-2xs font-mono uppercase tracking-label text-ink-tertiary">
                      candidate
                    </span>
                  )}
                </div>
                <p className="text-xs text-ink-secondary mt-1.5 ml-3.5 leading-relaxed">
                  {o.reason}
                </p>
                {o.available && (
                  <p className="text-2xs text-ink-tertiary mt-1 ml-3.5 leading-relaxed">
                    assumes: {o.assumption}
                  </p>
                )}
                {!!Object.keys(o.needs ?? {}).length && (
                  <p className="text-2xs font-mono text-ink-tertiary mt-1 ml-3.5">
                    needs {Object.entries(o.needs).map(([k, v]) => `${k} (${v})`).join(", ")}
                  </p>
                )}
              </button>
            );
          })}
        </div>
      </Pane>
      </div>

      <Pane caption={lane ? `configure ${lane}` : "configure"}>
        {!lane ? (
          <p className="text-xs text-ink-tertiary leading-relaxed">
            Pick a design on the left. Availability is a fact about the data; which one to
            use is a judgement about the world, and that one is yours.
          </p>
        ) : (
          <div className="space-y-3">
            {(PARAMS[lane] ?? []).map((p) => (
              <label key={p} className="block space-y-1">
                <Label>{p}</Label>
                <input
                  className="w-full bg-canvas-inset border border-edge-subtle px-2.5 py-1.5 text-xs font-mono text-ink placeholder:text-ink-tertiary focus:outline-none focus:border-edge-strong"
                  value={args[p] ?? ""}
                  placeholder={p === "covariates" ? "comma separated, optional" : "column name"}
                  onChange={(e) => setArgs({ ...args, [p]: e.target.value })}
                />
              </label>
            ))}
            <div className="pt-2 flex items-center gap-3">
              <Button tone="primary" onClick={go} disabled={!ready || busy}>
                {busy ? "running…" : "estimate"}
              </Button>
              <span className="text-2xs font-mono text-ink-tertiary">
                {ready ? "ready" : "fill the required columns"}
              </span>
            </div>
            <p className="text-2xs text-ink-tertiary leading-relaxed pt-2 border-t border-edge-subtle">
              {menu.find((m) => m.lane === lane)?.assumption}
            </p>
            {!!Object.keys(job.roles ?? {}).length && (
              <div className="pt-2 space-y-1">
                <Label>columns kept out of the adjustment</Label>
                {Object.values(job.roles)
                  .filter((r) => r.role === "mediator" || r.role === "collider" ||
                                 r.role === "proxy_for_outcome")
                  .slice(0, 5)
                  .map((r) => (
                    <p key={r.column} className="text-2xs text-ink-tertiary leading-relaxed">
                      <span className="font-mono text-ink-secondary">{r.column}</span>{" "}
                      is a {r.role.replace(/_/g, " ")} — {r.why}
                    </p>
                  ))}
              </div>
            )}
          </div>
        )}
      </Pane>
    </div>
  );
}
