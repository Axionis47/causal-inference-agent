import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ContextCitation, ContextColumnView, ContextDatasetView, ContextStatement, ContextView } from "../types";

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));
const groups = {
  Meaning: ["definition", "concept", "entity"],
  Values: ["units", "codes", "valid_ranges", "missing_codes"],
  Measurement: ["method", "measurement_source", "determined_by", "collection_time", "reference_period"],
  Construction: ["construction", "derivation"],
  Missingness: ["missingness"],
  Qualifications: ["qualifications"],
} as const;

function Statements({ values, cite }: { values: ContextStatement[]; cite: (c: ContextCitation) => void }) {
  return (
    <>
      {values.map((s) => (
        <div key={s.id} className="context-statement">
          {"field" in s && <strong>{s.field.replaceAll("_", " ")}</strong>}
          <p style={{ whiteSpace: "pre-wrap" }}>{s.value}</p>
          <small>
            {s.status} ·{" "}
            {s.source.kind === "extraction"
              ? s.basis === "interpretation"
                ? "Interpretation of supplied context"
                : "From supplied context"
              : "Updated by you"}
          </small>
          {s.qualifications && <p style={{ whiteSpace: "pre-wrap" }}>{s.qualifications}</p>}
          {s.source.text && <blockquote>{s.source.text}</blockquote>}
          {s.citations.map((c, i) => (
            <button key={i} onClick={() => cite(c)}>
              View source {i + 1}
            </button>
          ))}
        </div>
      ))}
    </>
  );
}

function Relations({
  values,
  columns,
  concepts,
  cite,
  openColumn,
}: {
  values: ContextColumnView["relationships"];
  columns: ContextDatasetView["pack"]["columns"];
  concepts: ContextDatasetView["concepts"];
  cite: (c: ContextCitation) => void;
  openColumn: (id: string) => void;
}) {
  return (
    <>
      {values.map((r) => (
        <div key={r.id} className="context-relation">
          <h4>{r.field.replaceAll("_", " ")}</h4>
          <Statements values={[r]} cite={cite} />
          <ul>
            {r.participants.map((p, i) => (
              <li key={i}>
                {p.role}:{" "}
                {p.kind === "column" ? (
                  <button onClick={() => openColumn(p.id)}>{columns.find((c) => c.column_id === p.id)?.name ?? p.id}</button>
                ) : (
                  (concepts.find((c) => c.id === p.id)?.value ?? p.id)
                )}
              </li>
            ))}
          </ul>
          {r.timing && <p>Timing: {r.timing}</p>}
          {r.conditions && <p>Conditions: {r.conditions}</p>}
          {r.directness !== "unspecified" && <p>Directness: {r.directness}</p>}
          {r.absence_scope !== "unspecified" && <p>Absence asserted: {r.absence_scope.replaceAll("_", " ")}</p>}
        </div>
      ))}
    </>
  );
}

function Column({
  datasetId,
  data,
  columnId,
  cite,
  openColumn,
}: {
  datasetId: string;
  data: ContextDatasetView;
  columnId: string;
  cite: (c: ContextCitation) => void;
  openColumn: (id: string) => void;
}) {
  const [result, setResult] = useState<ContextColumnView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    api
      .columnContext(datasetId, data.pack.revision_id, columnId)
      .then((v) => {
        if (active) setResult(v);
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, [datasetId, data.pack.revision_id, columnId, attempt]);
  if (error)
    return (
      <div role="alert">
        {error}
        <button
          onClick={() => {
            setError(null);
            setAttempt((a) => a + 1);
          }}
        >
          Retry column context
        </button>
      </div>
    );
  if (!result) return <p role="status">Loading column context…</p>;
  return (
    <div aria-label={`Context for ${result.column.name}`}>
      {Object.entries(groups).map(([label, fields]) => {
        const values = result.statements.filter((s) => (fields as readonly string[]).includes(s.field));
        return (
          <section key={label}>
            <h4>{label}</h4>
            {values.length ? <Statements values={values} cite={cite} /> : <p>Not supplied</p>}
          </section>
        );
      })}
      <h4>Relevant study context</h4>
      <Statements values={result.dataset} cite={cite} />
      <h4>Relationships</h4>
      {!result.relationships.length && <p>Not supplied</p>}
      <Relations values={result.relationships} columns={data.pack.columns} concepts={result.concepts} cite={cite} openColumn={openColumn} />
      <Statements values={result.concepts} cite={cite} />
    </div>
  );
}

function ContextPanel({ datasetId, packId, close }: { datasetId: string; packId: string; close: () => void }) {
  const [data, setData] = useState<ContextDatasetView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [opened, setOpened] = useState<string | null>(null);
  const [citation, setCitation] = useState<ContextCitation | null>(null);
  const sourceRef = useRef<HTMLDivElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  useEffect(() => {
    if (citation) sourceRef.current?.focus();
  }, [citation]);
  useEffect(() => {
    let active = true;
    api
      .datasetContext(datasetId, packId)
      .then((v) => {
        if (active) setData(v);
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, [datasetId, packId, attempt]);
  const chars = Array.from(data?.source_text ?? "");
  return (
    <aside
      className="context-panel"
      aria-label="Organized context"
      onKeyDown={(e) => {
        if (e.key === "Escape") close();
      }}
    >
      <button onClick={close}>Close context</button>
      <h2 tabIndex={-1} ref={headingRef}>
        Organized context
      </h2>
      {error && (
        <div role="alert">
          {error}
          <button
            onClick={() => {
              setError(null);
              setAttempt((a) => a + 1);
            }}
          >
            Retry context details
          </button>
        </div>
      )}
      {!data && !error && <p role="status">Loading context details…</p>}
      {data && (
        <>
          <p>
            Revision {data.pack.revision} · Updated {new Date(data.pack.created_at).toLocaleString()}
          </p>
          <p>
            Source version: <code>{data.pack.version_id}</code>
          </p>
          <h3>Dataset and study</h3>
          {!data.dataset.length && <p>Not supplied</p>}
          <Statements values={data.dataset} cite={setCitation} />
          {!!data.concepts.length && (
            <>
              <h3>Other concepts and events</h3>
              <Statements values={data.concepts} cite={setCitation} />
            </>
          )}
          <Relations values={data.relationships} columns={data.pack.columns} concepts={data.concepts} cite={setCitation} openColumn={setOpened} />
          <h3>Columns</h3>
          {data.pack.columns.map((c) => (
            <section className="profile-column" key={c.column_id}>
              <button aria-expanded={opened === c.column_id} onClick={() => setOpened(opened === c.column_id ? null : c.column_id)}>
                {c.name}
              </button>
              {opened === c.column_id && (
                <Column key={c.column_id} datasetId={datasetId} data={data} columnId={c.column_id} cite={setCitation} openColumn={setOpened} />
              )}
            </section>
          ))}
          {citation && (
            <div tabIndex={-1} ref={sourceRef} aria-label="Original source passage">
              <h3>Original study context</h3>
              <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
                {chars.slice(0, citation.start).join("")}
                <mark>{chars.slice(citation.start, citation.end).join("")}</mark>
                {chars.slice(citation.end).join("")}
              </p>
              <button onClick={() => setCitation(null)}>Close source passage</button>
            </div>
          )}
        </>
      )}
    </aside>
  );
}

export default function DatasetContext({ datasetId }: { datasetId: string }) {
  // Keyed inner state prevents old dataset data remaining visible even before effects run.
  return <ContextControl key={datasetId} datasetId={datasetId} />;
}

function ContextControl({ datasetId }: { datasetId: string }) {
  const [view, setView] = useState<ContextView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const ticket = useRef(0);
  const inFlight = useRef(false);
  const viewButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const n = ++ticket.current;
    api
      .context(datasetId)
      .then((v) => {
        if (n === ticket.current) setView(v);
      })
      .catch((e) => {
        if (n === ticket.current) setError(message(e));
      });
    return () => {
      ticket.current++;
    };
  }, [datasetId]);
  async function request(prepare: boolean) {
    if (inFlight.current) return;
    inFlight.current = true;
    const n = ++ticket.current;
    setBusy(true);
    setError(null);
    try {
      const v = await (prepare ? api.prepareContext(datasetId) : api.context(datasetId));
      if (n === ticket.current) {
        setView(v);
        if (prepare) setOpen(true);
      }
    } catch (e) {
      if (n === ticket.current) setError(message(e));
    } finally {
      inFlight.current = false;
      if (n === ticket.current) setBusy(false);
    }
  }
  return (
    <section aria-label="Dataset context">
      <h2>Dataset context</h2>
      <p>Current dataset and column context, including your corrections and original source citations.</p>
      {!view && !error && <p role="status">Loading saved context…</p>}
      {error && (
        <div role="alert">
          {error}
          <button disabled={busy} onClick={() => void request(false)}>
            Reload context
          </button>
        </div>
      )}
      {view && (
        <button disabled={busy} onClick={() => void request(!view.pack)}>
          {busy ? "Preparing context…" : view.pack ? "Refresh context" : "Prepare context"}
        </button>
      )}
      {busy && <p role="status">Arranging the supplied context…</p>}
      {view?.pack && (
        <button ref={viewButton} onClick={() => setOpen(true)}>
          View context
        </button>
      )}
      {open && view?.pack && (
        <ContextPanel
          key={view.pack.revision_id}
          datasetId={datasetId}
          packId={view.pack.revision_id}
          close={() => {
            setOpen(false);
            viewButton.current?.focus();
          }}
        />
      )}
    </section>
  );
}
