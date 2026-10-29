import { Link } from "react-router-dom";
import type { DatasetListItem } from "../types";

export default function DatasetCard({ d }: { d: DatasetListItem }) {
  return (
    <article className="card">
      <h2>
        <Link to={`/datasets/${d.dataset_id}`}>{d.title}</Link>
      </h2>
      <p>
        {d.filename} · {d.size_bytes.toLocaleString()} bytes
      </p>
      <p>Uploaded {new Date(d.created_at).toLocaleString()}</p>
      <p>Saved · {d.has_initial_context ? "Original context supplied" : "No original context"}</p>
      <Link className="btn sm" to={`/datasets/${d.dataset_id}`}>
        Open dataset
      </Link>
    </article>
  );
}
