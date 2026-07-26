/** Screen 1: pick a dataset, ask a question. Two fields, no wizard. */
import { useEffect, useState } from "react";
import { api, type Dataset } from "../api";
import { Button, Dot, Field, Label, Pane } from "../ui";

const SUGGESTED: Record<string, { q: string; c: string }> = {
  heart_failure: {
    q: "Does high blood pressure affect how long patients survive?",
    c: "Clinical records. DEATH_EVENT marks death during follow-up; time is days observed.",
  },
  lalonde: {
    q: "Did the job training program raise earnings in 1978?",
    c: "NSW training trial treated units, combined with a PSID comparison group.",
  },
  card: {
    q: "What is the effect of years of schooling on log hourly wages?",
    c: "US men in 1976. nearc4 marks growing up near a four-year college. Use lwage.",
  },
  card_krueger: {
    q: "Did New Jersey's minimum wage rise change fast-food employment?",
    c: "Stores in NJ and PA surveyed before and after. period 0 is before, 1 is after.",
  },
  ihdp: {
    q: "What is the effect of the intervention on the child's test score?",
    c: "Infant Health and Development Program. y_factual is the observed score.",
  },
  bank: {
    q: "Does the higher recovery strategy increase the amount recovered?",
    c: "Customers above an expected recovery of 1000 get a more intensive strategy.",
  },
  student: {
    q: "Does study time affect final grades through past failures?",
    c: "Portuguese secondary school records. G3 is the final grade.",
  },
  visitors: {
    q: "Did anything change site traffic at the start of 2018?",
    c: "Daily website statistics. Unique.Visits is the visitor count.",
  },
};

export function Ask({ onStarted }: { onStarted: (id: string) => void }) {
  const [sets, setSets] = useState<Dataset[]>([]);
  const [picked, setPicked] = useState("");
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.datasets().then(setSets).catch((e) => setError(String(e)));
  }, []);

  function choose(name: string) {
    setPicked(name);
    const s = SUGGESTED[name];
    if (s) {
      setQuestion(s.q);
      setContext(s.c);
    }
  }

  async function start() {
    setBusy(true);
    setError("");
    try {
      const { id } = await api.createJob(picked, question, context);
      onStarted(id);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  return (
    <div className="grid grid-cols-[320px_1fr] gap-4 h-full">
      <Pane caption="datasets" right={<span className="text-2xs font-mono text-ink-tertiary tabular">{sets.length}</span>}>
        <div className="space-y-0.5">
          {sets.map((d) => (
            <button
              key={d.name}
              onClick={() => choose(d.name)}
              className={`w-full flex items-center gap-2 px-2 py-1.5 text-left border transition-colors ${
                picked === d.name
                  ? "border-amber-dim bg-canvas-overlay"
                  : "border-transparent hover:bg-canvas-overlay"
              }`}
            >
              <Dot status={picked === d.name ? "live" : "pending"} />
              <span className="text-xs font-mono text-ink flex-1">{d.name}</span>
              <span className="text-2xs font-mono text-ink-tertiary tabular">
                {d.n_columns}c
              </span>
            </button>
          ))}
          {!sets.length && (
            <p className="text-xs text-ink-tertiary py-4 text-center">loading…</p>
          )}
        </div>
      </Pane>

      <Pane caption="question">
        <div className="space-y-4 max-w-2xl">
          <Field
            label="what do you want to know"
            value={question}
            onChange={setQuestion}
            rows={3}
            placeholder="Does X affect Y?"
          />
          <Field
            label="what should we know about this data"
            value={context}
            onChange={setContext}
            rows={3}
            placeholder="Anything that explains what the columns mean."
          />
          <div className="flex items-center gap-3 pt-1">
            <Button tone="primary" onClick={start} disabled={!picked || !question || busy}>
              {busy ? "starting…" : "run"}
            </Button>
            {picked && (
              <span className="text-2xs font-mono text-ink-tertiary">
                against {picked}
              </span>
            )}
          </div>
          {error && (
            <p className="text-xs font-mono text-rose flex items-center gap-2">
              <Dot status="failed" /> {error}
            </p>
          )}
          <p className="text-xs text-ink-tertiary pt-2 leading-relaxed">
            The engine reads your question against the column list, then shows which
            designs this data can support. It does not choose one. <Label>you do</Label>
          </p>
        </div>
      </Pane>
    </div>
  );
}
