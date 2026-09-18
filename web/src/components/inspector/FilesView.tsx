import { api } from "../../api";
import type { Selection } from "../../selection";
import type { SessionView } from "../../types";
import FileViewer, { viewable } from "./FileViewer";

export default function FilesView({ view, run, file, onSelect }: { view: SessionView; run: number | undefined; file: string | undefined; onSelect: (s: Selection) => void }) {
  const written = view.written;
  const own = written ? ["note", "claims", "profile"].filter((k) => written[k]) : [];
  if (!view.runs.length && !own.length) return <p className="empty-note">No files yet. The dataset's own appear once the description is read; each run adds its report and tables.</p>;
  const r = view.runs.find((x) => x.index === run) ?? view.runs[view.runs.length - 1];
  return (
    <>
      {own.length > 0 && (
        <section>
          <h3>Dataset files</h3>
          <div className="mono muted small">
            {own.map((k) => (
              <div key={k}>{String(written![k])}</div>
            ))}
          </div>
        </section>
      )}
      {r && (
        <section className="files-run">
          <div className="picker" role="tablist" aria-label="Runs">
            {view.runs.map((x) => (
              <button key={x.index} type="button" role="tab" aria-selected={x.index === r.index} className={`chip${x.index === r.index ? " on" : ""}`} onClick={() => onSelect({ tab: "files", run: x.index })}>
                Run {x.index}
              </button>
            ))}
          </div>
          {!r.run_id ? (
            <p className="muted">Run {r.index} left no files on disk.</p>
          ) : (
            <div className="chips">
              {r.files.map((f) =>
                viewable(f) ? (
                  <button key={f} type="button" className={`chip${file === f ? " on" : ""}`} onClick={() => onSelect({ tab: "files", run: r.index, file: f })}>
                    {f}
                  </button>
                ) : (
                  <a key={f} className="chip" href={api.fileUrl(r.run_id!, f)} download>
                    {f} ↓
                  </a>
                ),
              )}
            </div>
          )}
        </section>
      )}
      {r?.run_id && file && (viewable(file) ? <FileViewer runId={r.run_id} name={file} /> : (
        <p className="muted">
          {file} is a table.{" "}
          <a href={api.fileUrl(r.run_id, file)} download>
            Download it
          </a>
          .
        </p>
      ))}
    </>
  );
}
