import { Link } from "react-router-dom";
import type { DatasetSummary } from "../types";
import StagePill from "./StagePill";

export default function DatasetCard({ d, onDelete }: { d: DatasetSummary; onDelete: (d: DatasetSummary) => void }) {
  return (
    <article className="card">
      <h2 className="title">
        <Link to={`/d/${encodeURIComponent(d.name)}`}>{d.title}</Link>
      </h2>
      <div className="meta">
        <span>{d.name}</span>
        {d.rows != null && (
          <span>
            {d.rows.toLocaleString()} rows × {d.columns} columns
          </span>
        )}
        <StagePill d={d} />
      </div>
      {d.question ? <p className="q">{d.question}</p> : <p className="q muted">No question recorded.</p>}
      <div className="foot">
        <Link className="btn sm" to={`/d/${encodeURIComponent(d.name)}`}>
          Open
        </Link>
        <button className="btn quiet sm danger" onClick={() => onDelete(d)}>
          Delete
        </button>
      </div>
    </article>
  );
}
