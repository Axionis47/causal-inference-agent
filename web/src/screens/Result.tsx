/** Screen 5: one number is the hero, everything else supports it.
 *
 *  Two rules from the spec that the code enforces rather than trusts:
 *    - the interval is always shown as an interval, never a bare point estimate
 *    - claim strength comes from the server's lookup, so the UI cannot overclaim
 */
import { api, type Result as R } from "../api";
import { Dot, Label, Pane, Row, type Status } from "../ui";

const STRENGTH: Record<string, { tone: string; dot: Status; says: string }> = {
  moderate: {
    tone: "text-mint",
    dot: "ok",
    says: "the evidence supports an effect, under this design's assumptions",
  },
  weak: {
    tone: "text-amber",
    dot: "info",
    says: "an adjusted association; confounding remains a live explanation",
  },
  none: {
    tone: "text-ink-secondary",
    dot: "pending",
    says: "the interval covers the null, so no clear effect is shown",
  },
};

export function Result({ result }: { result: R }) {
  const e = result.estimate;
  const s = STRENGTH[result.strength ?? "none"] ?? STRENGTH.none;

  if (result.status === "failed" || !e) {
    return (
      <Pane caption="failed">
        <div className="flex items-start gap-2">
          <Dot status="failed" />
          <p className="text-xs font-mono text-rose leading-relaxed">
            {result.error ?? "no estimate was produced"}
          </p>
        </div>
        <p className="text-xs text-ink-tertiary mt-3 leading-relaxed">
          A design refusing is a result, not a crash. The message above names the
          condition the data did not meet.
        </p>
      </Pane>
    );
  }

  const interval =
    e.ci_low !== null && e.ci_high !== null
      ? `${num(e.ci_low)} to ${num(e.ci_high)}`
      : "not available";

  return (
    <div className="grid grid-cols-[1fr_340px] gap-4 h-full min-h-0">
      <div className="space-y-4 min-h-0 overflow-auto max-h-[calc(100vh-160px)] pr-1">
        <Pane caption="estimate">
          <div className="space-y-3">
            <div className="flex items-baseline gap-3">
              <span className="text-2xl font-mono tabular text-ink">{num(e.value)}</span>
              <span className="text-2xs font-mono uppercase tracking-label text-ink-tertiary">
                {e.estimand.replace(/_/g, " ")}
              </span>
            </div>
            {/* the interval, always: a point estimate alone overstates precision */}
            <div className="flex items-center gap-2">
              <span className="text-2xs font-mono uppercase tracking-label text-ink-tertiary">
                95% interval
              </span>
              <span className="text-sm font-mono tabular text-ink-secondary">{interval}</span>
            </div>
            <div className="flex items-center gap-2 pt-1">
              <Dot status={s.dot} />
              <span className={`text-xs font-mono uppercase tracking-label ${s.tone}`}>
                {result.strength}
              </span>
              <span className="text-xs text-ink-secondary">— {s.says}</span>
            </div>
          </div>
        </Pane>

        {result.narrative && (
          <Pane caption="readout">
            <div className="space-y-3 max-w-2xl">
              {result.narrative.split(/\n\n+/).map((p, i) => (
                <p key={i} className="text-sm text-ink-secondary leading-relaxed">
                  {p}
                </p>
              ))}
            </div>
          </Pane>
        )}
      </div>

      <div className="space-y-4">
        <Pane caption="method">
          <div className="space-y-0.5">
            <Row k="design" v={result.lane ?? "—"} />
            <Row k="estimator" v={e.estimator} />
            <Row k="rows used" v={e.n.toLocaleString()} />
            {e.se !== null && <Row k="std. error" v={num(e.se)} />}
            {e.p_value !== null && (
              <Row
                k="p-value"
                v={e.p_value < 0.001 ? "<0.001" : num(e.p_value)}
                tone={e.p_value < 0.05 ? "text-mint" : "text-ink-secondary"}
              />
            )}
          </div>
        </Pane>

        {!!e.notes.length && (
          <Pane caption="caveats">
            <ul className="space-y-2">
              {e.notes.map((n, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="mt-1.5">
                    <Dot status="info" />
                  </span>
                  <span className="text-xs text-ink-secondary leading-relaxed">{n}</span>
                </li>
              ))}
            </ul>
          </Pane>
        )}

        <Pane caption="reproduce">
          <a
            href={api.notebookUrl(result.id)}
            download={`${result.id}.ipynb`}
            className="inline-block px-3 py-1.5 text-2xs font-mono uppercase tracking-label border border-edge text-ink-secondary hover:text-ink hover:border-edge-strong"
          >
            download notebook
          </a>
          <p className="text-xs text-ink-tertiary leading-relaxed mt-2">
            It recomputes this estimate by calling the same function against the
            same file, then asserts the answer matches what you see here. Running
            it checks the tool rather than restating it.
          </p>
        </Pane>

        <Pane caption="honesty">
          <p className="text-xs text-ink-tertiary leading-relaxed">
            Claim strength is a lookup, not an opinion. The design sets the ceiling and an
            interval covering the null lowers it. No observational design reaches{" "}
            <Label>strong</Label>
          </p>
        </Pane>
      </div>
    </div>
  );
}

function num(n: number): string {
  if (n !== 0 && Math.abs(n) < 0.001) return n.toExponential(2);
  if (Math.abs(n) >= 10000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return String(Math.round(n * 10000) / 10000);
}
