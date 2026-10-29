import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import TopBar from "../components/TopBar";
import UploadDrop from "../components/UploadDrop";
import { useDatasets } from "../components/Shell";

export default function NewDataset() {
  const navigate = useNavigate();
  const { refresh } = useDatasets();
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [context, setContext] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const key = useRef(crypto.randomUUID());
  const submitting = useRef(false);
  const changed = () => {
    key.current = crypto.randomUUID();
  };

  const submit = async () => {
    if (!file || !title.trim() || submitting.current) return;
    submitting.current = true;
    setBusy(true);
    setError(null);
    setProgress(0);
    try {
      const result = await api.ingest(file, title.trim(), context, key.current, setProgress);
      void refresh();
      navigate(`/datasets/${result.dataset_id}`, { state: { saved: true } });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  };

  return (
    <div className="wrap">
      <TopBar
        crumb={
          <>
            <Link to="/">Datasets</Link> / upload
          </>
        }
      />
      <section className="hero">
        <div>
          <h1>Upload a dataset</h1>
          <p className="lede">Save your original CSV and study context. Profiling and conversations are not started by this step.</p>
        </div>
      </section>
      <form
        className="form"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <fieldset disabled={busy} style={{ border: 0, padding: 0 }}>
          <section className="stage">
            <label className="field">
              <span>Dataset title</span>
              <input
                className="input"
                required
                maxLength={200}
                value={title}
                onChange={(e) => {
                  setTitle(e.target.value);
                  changed();
                }}
              />
            </label>
          </section>
          <section className="stage">
            <h2>CSV file</h2>
            <UploadDrop
              busy={busy}
              onFile={(f) => {
                setFile(f);
                changed();
              }}
            />
            {file && (
              <p aria-live="polite">
                Selected: {file.name} ({file.size.toLocaleString()} bytes)
              </p>
            )}
            <p className="help">UTF-8 CSV with unique column headers and at least one data row. Default maximum: 25 MiB.</p>
          </section>
          <section className="stage">
            <label className="field">
              <span>Original study context (optional)</span>
              <textarea
                className="input"
                rows={7}
                maxLength={100000}
                value={context}
                onChange={(e) => {
                  setContext(e.target.value);
                  changed();
                }}
              />
            </label>
            <p className="help">
              Describe where this data came from and what was happening when it was collected. Your original account will be saved unchanged.
            </p>
          </section>
          <button className="btn primary" type="submit" disabled={!file || !title.trim() || busy}>
            {busy ? "Saving…" : "Save dataset"}
          </button>
        </fieldset>
        {busy && (
          <div role="status">
            <progress max={100} value={progress} aria-label="Upload progress" />
            <p>{progress < 100 ? `Uploading: ${progress}%` : "Upload received. Validating and saving…"}</p>
          </div>
        )}
      </form>
      {error && (
        <p className="err" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
