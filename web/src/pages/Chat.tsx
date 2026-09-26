import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Link, useParams } from "react-router-dom";
import Activity from "../components/Activity";
import Composer from "../components/Composer";
import ConfirmDialog from "../components/ConfirmDialog";
import Inspector from "../components/Inspector";
import { useDatasets } from "../components/Shell";
import StatusStrip from "../components/StatusStrip";
import TabStrip, { type Counts } from "../components/TabStrip";
import TopBar from "../components/TopBar";
import Transcript from "../components/Transcript";
import { usePaneResize } from "../hooks/usePaneResize";
import { useSelection } from "../hooks/useSelection";
import { resolveSelection } from "../selection";
import { useSession } from "../store";

export default function Chat() {
  const { name = "" } = useParams();
  const { view, error, sending, send, resume, newAnalysis } = useSession(name);
  const { refresh } = useDatasets();
  const { sel: raw, set: select, close } = useSelection();
  const [ending, setEnding] = useState(false);
  const [asking, setAsking] = useState(false);
  const deskRef = useRef<HTMLDivElement>(null);
  const { width, splitterProps } = usePaneResize(deskRef);

  // keep the sidebar's pill for this dataset honest as the conversation moves
  const stage = view?.stage;
  const runCount = view?.runs.length ?? 0;
  useEffect(() => {
    void refresh();
  }, [refresh, stage, runCount]);

  if (!view) {
    return (
      <div className="wrap">
        <TopBar crumb={<Link to="/">Datasets</Link>} />
        {error ? (
          <p className="err">{error}</p>
        ) : (
          <p className="muted" style={{ padding: "40px 0" }}>
            Loading…
          </p>
        )}
      </div>
    );
  }

  const sel = resolveSelection(raw, view.runs);
  const busy = view.stage === "busy" || sending;
  const canTalk = view.stage === "waiting" && !sending;
  const openClaims = view.status?.open.length ?? 0;
  const fileCount = view.runs.reduce((n, r) => n + r.files.length, 0);
  const endWord = view.phase === "after" ? "done" : "quit";
  const counts: Counts = {
    claims: view.status ? (openClaims ? `${openClaims} open` : "settled") : view.claims.length ? String(view.claims.length) : undefined,
    runs: view.runs.length ? String(view.runs.length) : undefined,
    files: fileCount ? String(fileCount) : undefined,
  };

  return (
    <div className="desk" ref={deskRef} data-pane={sel ? "open" : "closed"} style={{ "--pane-pref": width ? `${width}px` : undefined } as CSSProperties}>
      <TopBar
        crumb={
          <>
            <Link to="/">Datasets</Link> / {view.name}
          </>
        }
        right={
          <span className="pill">
            {view.stage === "busy"
              ? "working"
              : view.stage === "waiting"
                ? view.phase === "after"
                  ? `after run ${view.runs.length}`
                  : "interviewing"
                : view.stage}
          </span>
        }
      />
      <div className="chat-head">
        <div>
          <h1>{view.title}</h1>
          {view.question && <p className="q">{view.question}</p>}
        </div>
      </div>
      <main>
        <Transcript turns={view.transcript} />
        {error && <p className="err">{error}</p>}
        {view.stage === "stale" && (
          <div className="banner warn">
            <span>The last step did not finish; the server stopped mid-way. Continue from where it left off.</span>
            <button className="btn sm" onClick={resume}>
              Resume
            </button>
          </div>
        )}
        {view.stage === "error" && (
          <div className="banner bad">
            <span>{view.error}</span>
            <button className="btn sm" onClick={resume}>
              Try again
            </button>
          </div>
        )}
        {view.stage === "ended" && (
          <div className="banner">
            <span>This conversation ended. Ask a new question of the same file; what is known about it carries over.</span>
            <button className="btn sm" onClick={newAnalysis}>
              New question
            </button>
          </div>
        )}
        {view.stage === "new" && (
          <div className="banner">
            <span>Not started yet.</span>
            <button className="btn sm" onClick={newAnalysis}>
              Start
            </button>
          </div>
        )}
        {(view.stage === "waiting" || view.stage === "busy") && (
          <Composer view={view} disabled={!canTalk} onSend={send} onEnd={() => setEnding(true)} onNew={() => setAsking(true)} />
        )}
        {busy && view.activity && <Activity node={view.activity.node} />}
        {view.phase === "before" && <StatusStrip claims={view.claims} status={view.status} onOpen={() => select({ tab: "claims" })} />}
      </main>
      {sel ? (
        <Inspector view={view} sel={sel} counts={counts} width={width} splitterProps={splitterProps} onSelect={select} onClose={close} />
      ) : (
        <TabStrip counts={counts} onPick={(t) => select({ tab: t })} />
      )}
      <ConfirmDialog
        open={ending}
        title="End the conversation?"
        body={
          view.phase === "after"
            ? "The runs and their files stay. You can start a new conversation on this dataset later."
            : "Nothing has been written for the analysis yet. The claims settled so far are kept in the conversation's checkpoint only."
        }
        confirmLabel="End conversation"
        onConfirm={() => {
          setEnding(false);
          send(endWord);
        }}
        onCancel={() => setEnding(false)}
      />
      <ConfirmDialog
        open={asking}
        title="Ask a new question of this file?"
        body={
          view.phase === "after"
            ? "This conversation closes and a new one starts on the same file. What is known about the columns carries over; the runs and their files stay listed."
            : "This conversation closes before a run. The claims settled so far carry over to the new question where they still apply."
        }
        confirmLabel="New question"
        onConfirm={() => {
          setAsking(false);
          newAnalysis();
        }}
        onCancel={() => setAsking(false)}
      />
    </div>
  );
}
