/** The shell: top bar, step rail, and whichever screen the run is up to.
 *
 *  Screen selection follows the job's status rather than local navigation
 *  state, so a reload lands you exactly where the run actually is.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, openTape, type Job, type Result as R, type TapeEvent } from "./api";
import { Ask } from "./screens/Ask";
import { Data } from "./screens/Data";
import { Design } from "./screens/Design";
import { Result } from "./screens/Result";
import { Running } from "./screens/Running";
import { Dot, Label, Steps } from "./ui";

type View = "ask" | "data" | "design" | "running" | "result";

export default function App() {
  const [jobId, setJobId] = useState<string>(() => localStorage.getItem("job") ?? "");
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<R | null>(null);
  const [tape, setTape] = useState<TapeEvent[]>([]);
  const [view, setView] = useState<View>("ask");
  const [seenData, setSeenData] = useState(false);
  const closeTape = useRef<null | (() => void)>(null);

  const poll = useCallback(async (id: string) => {
    try {
      const j = await api.job(id);
      setJob(j);
      if (j.status === "completed" || j.status === "failed") {
        setResult(await api.result(id));
      }
      return j;
    } catch {
      return null;
    }
  }, []);

  // one tape per job; it replays from the start so a reload loses nothing
  useEffect(() => {
    if (!jobId) return;
    localStorage.setItem("job", jobId);
    setTape([]);
    closeTape.current?.();
    closeTape.current = openTape(jobId, (e) => {
      setTape((t) => [...t, e]);
      if (e.event === "waiting_for_you" || e.event === "completed" || e.event === "failed") {
        poll(jobId);
      }
    });
    poll(jobId);
    const timer = setInterval(() => poll(jobId), 1500);
    return () => {
      clearInterval(timer);
      closeTape.current?.();
    };
  }, [jobId, poll]);

  // the run's status decides the screen, except while you are reading the data
  useEffect(() => {
    if (!job) return;
    if (job.status === "completed" || job.status === "failed") setView("result");
    else if (job.status === "waiting_for_you") setView(seenData ? "design" : "data");
    else setView("running");
  }, [job, seenData]);

  function reset() {
    closeTape.current?.();
    localStorage.removeItem("job");
    setJobId("");
    setJob(null);
    setResult(null);
    setTape([]);
    setSeenData(false);
    setView("ask");
  }

  async function reopen() {
    if (!jobId) return;
    await api.reopen(jobId);
    setSeenData(true); // straight back to the gate; the data was already checked
    setResult(null);
    setView("design");
    poll(jobId);
  }

  async function chooseDesign(lane: string, kwargs: Record<string, unknown>) {
    if (!jobId) return;
    await api.chooseDesign(jobId, lane, kwargs);
    setView("running");
    poll(jobId);
  }

  const step = { ask: 0, data: 1, design: 2, running: 3, result: 4 }[view];
  const live = job?.status === "running";

  return (
    <div className="h-full flex flex-col bg-canvas">
      <header className="flex items-center justify-between px-4 py-2.5 border-b border-edge-subtle bg-canvas-raised shrink-0">
        <div className="flex items-center gap-4">
          <span className="text-2xs font-mono uppercase tracking-label text-ink">
            causal engine
          </span>
          {jobId && (
            <span className="flex items-center gap-2">
              <Dot status={live ? "live" : job?.status === "failed" ? "failed" : "ok"} />
              <span className="text-2xl font-mono tabular text-ink">{jobId}</span>
              <span className="text-2xs font-mono uppercase tracking-label text-ink-tertiary">
                {job?.status?.replace(/_/g, " ") ?? "…"}
              </span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-6">
          <Steps at={step} />
          {jobId && (
            <button
              onClick={reset}
              className="text-2xs font-mono uppercase tracking-label text-ink-tertiary hover:text-ink"
            >
              new run
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 p-4 min-h-0">
        {view === "ask" && <Ask onStarted={setJobId} />}
        {view === "data" && job && (
          <div className="h-full flex flex-col gap-3 min-h-0">
            <div className="flex-1 min-h-0">
              <Data job={job} />
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <button
                onClick={() => setSeenData(true)}
                className="px-3 py-1.5 text-2xs font-mono uppercase tracking-label border border-amber-dim text-amber hover:bg-amber hover:text-ink-inverse"
              >
                looks right, choose a design
              </button>
              <span className="text-2xs font-mono text-ink-tertiary">
                check the reading before any maths happens
              </span>
            </div>
          </div>
        )}
        {view === "design" && job && <Design job={job} onChosen={chooseDesign} />}
        {view === "running" && <Running tape={tape} done={!live} />}
        {view === "result" && result && <Result result={result} onReopen={reopen} />}
        {view === "result" && !result && (
          <p className="text-xs text-ink-tertiary">loading result…</p>
        )}
      </main>

      <footer className="px-4 py-2 border-t border-edge-subtle shrink-0">
        <span className="text-2xs text-ink-tertiary">
          code says which designs are possible. <Label>you say which to use</Label>
        </span>
      </footer>
    </div>
  );
}
