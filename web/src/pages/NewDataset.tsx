import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError } from "../api";
import ColumnForm from "../components/ColumnForm";
import DatasetPreview from "../components/DatasetPreview";
import { useDatasets } from "../components/Shell";
import TopBar from "../components/TopBar";
import UploadDrop from "../components/UploadDrop";
import type { ProfileOut } from "../types";

const NAME_RE = /^[a-z][a-z0-9_]{1,39}$/;

export function slug(title: string): string {
  const s = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/^[^a-z]+/, "");
  return s.slice(0, 40);
}

export default function NewDataset() {
  const nav = useNavigate();
  const { refresh } = useDatasets();
  const [prof, setProf] = useState<ProfileOut | null>(null);
  const [reading, setReading] = useState(false);
  const [title, setTitle] = useState("");
  const [name, setName] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [question, setQuestion] = useState("");
  const [about, setAbout] = useState("");
  const [changed, setChanged] = useState("");
  const [cols, setCols] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const onFile = async (f: File) => {
    setReading(true);
    setError(null);
    try {
      const p = await api.profile(f);
      setProf(p);
      if (!title) {
        const t = f.name.replace(/\.csv$/i, "").replace(/[_-]+/g, " ");
        setTitle(t);
        if (!nameTouched) setName(slug(t));
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setReading(false);
    }
  };

  const nameOk = NAME_RE.test(name);
  const described = prof ? prof.columns.filter((c) => (cols[c.name] ?? "").trim()).length : 0;
  const canCreate = !!prof && nameOk && title.trim() && question.trim() && about.trim() && changed.trim() && !creating;

  const create = async () => {
    if (!prof || !canCreate) return;
    setCreating(true);
    setError(null);
    try {
      const d = await api.createDataset({
        name,
        title: title.trim(),
        upload_id: prof.upload_id,
        question: question.trim(),
        about,
        changed,
        columns: prof.columns.map((c) => ({ name: c.name, description: cols[c.name] ?? "" })),
      });
      await refresh();
      nav(`/d/${encodeURIComponent(d.name)}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setCreating(false);
    }
  };

  return (
    <div className="wrap">
      <TopBar crumb={<><Link to="/">Datasets</Link> / new</>} />
      <section className="hero">
        <div>
          <h1>New dataset</h1>
          <p className="lede">Upload the table, then say in your own words what each column records and what changed. The conversation starts from that.</p>
        </div>
      </section>
      <div className="form">
        <section className="stage">
          <h2>The file</h2>
          {!prof ? (
            <UploadDrop onFile={onFile} busy={reading} />
          ) : (
            <div className="filecard">
              <span className="name">{prof.filename}</span>
              <span className="muted mono">
                {prof.rows.toLocaleString()} rows · {prof.columns.length} columns
              </span>
              <button className="btn quiet sm" style={{ marginLeft: "auto" }} onClick={() => setProf(null)}>
                Choose another
              </button>
            </div>
          )}
        </section>

        {prof && (
          <section className="stage">
            <h2>What the profiler read</h2>
            <p className="help">The first rows as they are, then each column's shape and anything odd in it. Check this before describing the columns.</p>
            <DatasetPreview profile={prof} />
          </section>
        )}

        {prof && (
          <>
            <section className="stage">
              <h2>What this is</h2>
              <p className="help">A title for the card, a short name for the files, and the question you want answered.</p>
              <div className="two">
                <label className="field">
                  <span>Title</span>
                  <input
                    className="input"
                    value={title}
                    onChange={(e) => {
                      setTitle(e.target.value);
                      if (!nameTouched) setName(slug(e.target.value));
                    }}
                  />
                </label>
                <label className="field">
                  <span>Name</span>
                  <input
                    className="input mono"
                    value={name}
                    onChange={(e) => {
                      setNameTouched(true);
                      setName(e.target.value);
                    }}
                    aria-invalid={!nameOk}
                  />
                  <small>{nameOk ? "Lowercase letters, digits and underscores; used for the files." : "Start with a letter; lowercase letters, digits and underscores only; 2 to 40 characters."}</small>
                </label>
              </div>
              <label className="field">
                <span>The causal question</span>
                <input className="input" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Did completing the course raise math scores?" />
                <small>One cause, one outcome, in plain words.</small>
              </label>
            </section>

            <section className="stage">
              <h2>About the dataset</h2>
              <p className="help">What one row is, who is in the file and how they got there, and anything missing or odd.</p>
              <label className="field">
                <textarea className="input" value={about} onChange={(e) => setAbout(e.target.value)} placeholder="Each row is one student's results from the May 2026 exam at one school; every student who sat is included." />
              </label>
              <h2>What changed</h2>
              <p className="help">The programme, policy, or event the question is about: what it was, who it reached, when, and how it was decided who got it.</p>
              <label className="field">
                <textarea className="input" value={changed} onChange={(e) => setChanged(e.target.value)} placeholder="A six-week prep course before the exam. Places were offered first to free-lunch students, then to anyone who asked." />
              </label>
            </section>

            <section className="stage">
              <h2>About each column</h2>
              <p className="help">
                One line each: what it records and whether it was fixed before the change, set by it, or measured after. {described} of {prof.columns.length} described.
              </p>
              <ColumnForm columns={prof.columns} values={cols} onChange={(n, v) => setCols((c) => ({ ...c, [n]: v }))} />
            </section>

            <section className="stage">
              <div className="actions">
                <button className="btn primary" disabled={!canCreate} onClick={create}>
                  {creating ? "Creating…" : "Create dataset and start"}
                </button>
                <span className="muted">The conversation opens on the next page.</span>
              </div>
              {error && <p className="err">{error}</p>}
            </section>
          </>
        )}
        {!prof && error && <p className="err">{error}</p>}
      </div>
    </div>
  );
}
