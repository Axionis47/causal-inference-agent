import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError } from "../api";
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
  const canCreate = !!prof && nameOk && title.trim() && !creating;

  const create = async () => {
    if (!prof || !canCreate) return;
    setCreating(true);
    setError(null);
    try {
      const d = await api.createDataset({ name, title: title.trim(), upload_id: prof.upload_id });
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
          <p className="lede">Upload the table. The conversation starts by asking what you want to know; everything about the data is settled in it, one question at a time.</p>
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
              <p className="help">A title for the card and a short name for the files.</p>
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
            </section>

            <section className="stage">
              <div className="actions">
                <button className="btn primary" disabled={!canCreate} onClick={create}>
                  {creating ? "Creating…" : "Create dataset and start"}
                </button>
                <span className="muted">The conversation opens on the next page and starts with your question.</span>
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
