import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { components } from "../generated/schema";
import { api } from "../api";
import CitedText from "../components/CitedText";
import TopBar from "../components/TopBar";
import ArtifactCard from "../components/ArtifactCard";
import AnalysisCard from "../components/AnalysisCard";
import AddContextStatement from "../components/AddContextStatement";
import ContextStatements from "../components/ContextStatements";

type Wire<K extends keyof components["schemas"]> = components["schemas"][K];
type Command = Wire<"ChatCommand">;

function GapAnswer({ question, disabled, send }: { question: Wire<"GapQuestion">; disabled: boolean; send: (body: Partial<Command>) => Promise<boolean> }) {
  const [text, setText] = useState("");
  return (
    <fieldset disabled={disabled}>
      <legend>{question.text}</legend>
      {question.options?.map((option) => (
        <button key={option.id} onClick={() => void send({ kind: "answer", question_id: question.id, option_id: option.id })}>
          {option.label}
        </button>
      ))}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send({ kind: "answer", question_id: question.id, text });
        }}
      >
        <label>
          Your answer
          <input value={text} onChange={(e) => setText(e.target.value)} />
        </label>
        <button disabled={!text.trim()}>Save answer</button>
      </form>
    </fieldset>
  );
}

export default function Chat() {
  const { name = "" } = useParams();
  const [view, setView] = useState<Wire<"ChatOverview"> | null>(null);
  const [messages, setMessages] = useState<Wire<"ChatMessage">[]>([]);
  const [before, setBefore] = useState<string | null>(null);
  const [context, setContext] = useState<Wire<"ChatContextSummary"> | null>(null);
  const [changes, setChanges] = useState<Wire<"ContextChanges"> | null>(null);
  const [statements, setStatements] = useState<Wire<"KnowledgeStatement">[]>([]);
  const [products, setProducts] = useState<Wire<"ArtifactSummary">[]>([]);
  const [analyses, setAnalyses] = useState<Wire<"AnalysisSummary">[]>([]);
  const [tab, setTab] = useState("Context");
  const [earlier, setEarlier] = useState(false);
  const [section, setSection] = useState("dataset");
  const [nextStatements, setNextStatements] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);
  const retry = useRef<Command | null>(null);
  useEffect(() => {
    let active = true;
    api
      .session(name)
      .then((v) => {
        if (active) setView(v);
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, [name]);
  useEffect(() => {
    if (view?.status !== "running") return;
    let active = true;
    const timer = setTimeout(
      () =>
        void api
          .session(name)
          .then((v) => {
            if (active) setView(v);
          })
          .catch((e) => {
            if (active) setError(String(e));
          }),
      1500,
    );
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [view, name]);
  useEffect(() => {
    let active = true;
    api
      .messages(name)
      .then((page) => {
        if (active) {
          setMessages([...page.items].reverse());
          setBefore(page.before);
        }
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, [name, view?.message_count]);
  useEffect(() => {
    let active = true;
    api
      .chatContext(name)
      .then((v) => {
        if (active) setContext(v);
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, [name, view?.revision_id]);
  useEffect(() => {
    let active = true;
    if (tab === "Context") {
      setStatements([]);
      api
        .chatStatements(name, section)
        .then((v) => {
          if (active) {
            setStatements(v.statements);
            setNextStatements(v.next_offset);
          }
        })
        .catch((e) => {
          if (active) setError(String(e));
        });
    }
    if (tab === "Artifacts")
      api
        .artifacts(name, earlier)
        .then((v) => {
          if (active) setProducts(v);
        })
        .catch((e) => {
          if (active) setError(String(e));
        });
    if (tab === "Analyses")
      api
        .analyses(name)
        .then((v) => {
          if (active) setAnalyses(v);
        })
        .catch((e) => {
          if (active) setError(String(e));
        });
    return () => {
      active = false;
    };
  }, [tab, section, name, earlier, view?.revision_id, view?.message_count]);
  async function submit(command: Command) {
    retry.current = command;
    setSending(true);
    setError("");
    try {
      setView(await api.command(name, command));
      retry.current = null;
      setText("");
      return true;
    } catch (e) {
      setError(String(e));
      return false;
    } finally {
      setSending(false);
    }
  }
  async function send(body: Partial<Command>) {
    if (view) return submit({ submission_id: crypto.randomUUID(), revision_id: view.revision_id, kind: "message", text: "", ...body });
    return false;
  }
  const disabled = sending || view?.status !== "idle";
  const column = context?.columns.find((c) => c.column_id === section);
  const focusLabel = view?.focus_label ?? "Selected reference";
  if (!view) return <div className="wrap">{error ? <p role="alert">{error}</p> : <p role="status">Loading chat…</p>}</div>;
  return (
    <div className="wrap">
      <TopBar crumb={<Link to={`/datasets/${view.dataset_id}`}>{view.title}</Link>} />
      <h1>{view.question}</h1>
      <p>
        Analysis readiness: {view.readiness.status.replaceAll("_", " ")}
        {view.readiness.missing ? `, ${view.readiness.missing} unresolved requirements` : ""}. Preparation never starts an analysis.
      </p>
      <div className="chat-workspace">
        <section aria-label="Conversation">
          {before && (
            <button
              onClick={async () => {
                const page = await api.messages(name, before);
                setMessages((old) => [...page.items].reverse().concat(old));
                setBefore(page.before);
              }}
            >
              Load earlier messages
            </button>
          )}
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <strong>{message.role === "assistant" ? "Desk" : "You"}</strong>
              <CitedText text={message.text} chat={name} />
            </article>
          ))}
          {view.focus && (
            <p role="status">
              Discussing {focusLabel}{" "}
              <button disabled={disabled} onClick={() => void send({ kind: "focus", references: [] })}>
                Clear focus
              </button>
            </p>
          )}
          {error && (
            <div role="alert">
              <p>{error}</p>
              {retry.current && (
                <button disabled={sending} onClick={() => void submit(retry.current!)}>
                  Retry same submission
                </button>
              )}
            </div>
          )}
          {view.error && <p role="alert">{view.error}</p>}
          {view.status === "interrupted" && (
            <button
              disabled={sending}
              onClick={() =>
                void api
                  .resume(name)
                  .then(setView)
                  .catch((e) => setError(String(e)))
              }
            >
              Resume interrupted turn
            </button>
          )}
          {view.status === "running" && <p role="status">Working on your request…</p>}
          {view.fast_track && view.questions.map((q) => <GapAnswer key={q.id} question={q} disabled={disabled} send={send} />)}
          {view.analysis_review && (
            <section className="stage">
              <h2>Analysis review</h2>
              <p>{view.analysis_review.question}</p>
              <p>{view.analysis_review.family}</p>
              <p>Assumptions: {view.analysis_review.assumptions}</p>
              <button disabled={disabled} onClick={() => void send({ kind: "authorize", snapshot_id: view.analysis_review!.id })}>
                Run analysis
              </button>
            </section>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void send({ kind: "message", text });
            }}
          >
            <label>
              Message
              <textarea value={text} onChange={(e) => setText(e.target.value)} disabled={disabled} />
            </label>
            <button disabled={disabled || !text.trim()}>Send</button>
          </form>
          <button disabled={disabled} onClick={() => void send({ kind: view.fast_track ? "pause" : "fast_track" })}>
            {view.fast_track ? "Pause fast-track" : view.questions.length ? "Continue fast-track" : "Fast-track analysis preparation"}
          </button>
        </section>
        <aside aria-label="Chat inspectors">
          <nav aria-label="Inspectors">
            {["Context", "Artifacts", "Analyses"].map((t) => (
              <button key={t} aria-pressed={t === tab} onClick={() => setTab(t)}>
                {t}
              </button>
            ))}
          </nav>
          {tab === "Context" && context && (
            <>
              <p>Selected context revision {context.revision}</p>
              <button
                onClick={() =>
                  void api
                    .contextChanges(name)
                    .then(setChanges)
                    .catch((e) => setError(String(e)))
                }
              >
                Review changes since this revision
              </button>
              {changes && (
                <section>
                  <h2>Context changes</h2>
                  {changes.items.length === 0 ? (
                    <p>No statement changes since this chat's revision.</p>
                  ) : (
                    changes.items.map((c, i) => (
                      <article key={i}>
                        <p>
                          Previously: {c.before?.value ?? "Not recorded"} ({c.before?.status ?? "absent"})
                        </p>
                        <p>
                          Now: {c.after?.value ?? "Withdrawn"} ({c.after?.status ?? "absent"})
                        </p>
                        <p>{c.after?.reason}</p>
                      </article>
                    ))
                  )}
                  {changes.next_offset !== null && (
                    <button
                      onClick={() =>
                        void api
                          .contextChanges(name, changes.next_offset!)
                          .then((v) => setChanges((old) => ({ ...v, items: [...(old?.items ?? []), ...v.items] })))
                          .catch((e) => setError(String(e)))
                      }
                    >
                      More changes
                    </button>
                  )}
                </section>
              )}
              <button disabled={disabled} onClick={() => void send({ kind: "refresh" }).then(() => setChanges(null))}>
                Refresh to latest reviewed context
              </button>
              <label>
                Inspect
                <select aria-label="Inspect" value={section} onChange={(e) => setSection(e.target.value)}>
                  <option value="dataset">Dataset context</option>
                  <option value="relationship">Relationships</option>
                  <option value="requirement">Analysis requirements</option>
                  {context.columns.map((c) => (
                    <option key={c.column_id} value={c.column_id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </label>
              {column && (
                <button disabled={disabled} onClick={() => void send({ kind: "focus", references: [{ kind: "column", id: column.column_id }] })}>
                  Discuss this column
                </button>
              )}
              <ContextStatements
                statements={statements}
                columns={context.columns}
                disabled={disabled}
                onEdit={(edits) => send({ kind: "correct", edits, text: "Apply the displayed context corrections." })}
              />
              {nextStatements !== null && (
                <button
                  onClick={() =>
                    void api
                      .chatStatements(name, section, nextStatements)
                      .then((v) => {
                        setStatements((old) => old.concat(v.statements));
                        setNextStatements(v.next_offset);
                      })
                      .catch((e) => setError(String(e)))
                  }
                >
                  Load more statements
                </button>
              )}
              <AddContextStatement
                key={section}
                section={section}
                columns={context.columns}
                disabled={disabled}
                save={(edits) => send({ kind: "correct", edits, text: "Save the supplied context statement." })}
              />
            </>
          )}
          {tab === "Artifacts" && (
            <>
              <button onClick={() => setEarlier(false)} aria-pressed={!earlier}>
                This chat
              </button>
              <button onClick={() => setEarlier(true)} aria-pressed={earlier}>
                Earlier products from this dataset
              </button>
              {!products.length && <p>No artifacts in this chat yet. Ask for a visualization.</p>}
              {products.map((a) => (
                <ArtifactCard
                  key={a.id}
                  chat={name}
                  item={a}
                  earlier={earlier}
                  disabled={disabled}
                  discuss={(id, element) => send({ kind: earlier ? "reuse" : "focus", references: [{ kind: "artifact", id, element }] })}
                />
              ))}
              <button
                disabled={products.length === 0}
                onClick={() =>
                  void api
                    .artifacts(name, earlier, products.length)
                    .then((v) => setProducts((old) => old.concat(v)))
                    .catch((e) => setError(String(e)))
                }
              >
                Load more artifacts
              </button>
            </>
          )}
          {tab === "Analyses" && (
            <>
              {!analyses.length && <p>No analysis has run yet.</p>}
              {analyses.map((r) => (
                <AnalysisCard key={r.id} chat={name} item={r} />
              ))}
              <button
                disabled={analyses.length === 0}
                onClick={() =>
                  void api
                    .analyses(name, analyses.length)
                    .then((v) => setAnalyses((old) => old.concat(v)))
                    .catch((e) => setError(String(e)))
                }
              >
                Load more analyses
              </button>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
