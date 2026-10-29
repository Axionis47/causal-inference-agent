import { Link, NavLink } from "react-router-dom";
import type { useDatasetLibrary } from "../hooks/useDatasetLibrary";

export default function Sidebar({
  library,
  current,
  open,
  onToggle,
  onNav,
}: {
  library: ReturnType<typeof useDatasetLibrary>;
  current: string | null;
  open: boolean;
  onToggle: () => void;
  onNav: () => void;
}) {
  return (
    <nav className={`side${open ? "" : " slim"}`} aria-label="Datasets">
      <div className="side-top">
        {open && (
          <Link to="/" className="brand" onClick={onNav}>
            <span className="word">Causal desk</span>
          </Link>
        )}
        <button className="btn quiet sm" onClick={onToggle} aria-expanded={open} title={open ? "Collapse the dataset list" : "Expand the dataset list"}>
          {open ? "Collapse" : "Expand"}
        </button>
      </div>
      <div className="side-new">
        <Link className={open ? "btn primary sm" : "btn quiet sm"} to="/new" onClick={onNav} title="New dataset">
          {open ? "New dataset" : "New"}
        </Link>
      </div>
      {open && (
        <ul className="side-list">
          {library.items?.map((d) => (
            <li key={d.dataset_id}>
              <NavLink
                to={`/datasets/${encodeURIComponent(d.dataset_id)}`}
                className={d.dataset_id === current ? "on" : undefined}
                aria-current={d.dataset_id === current ? "page" : undefined}
                onClick={onNav}
              >
                <span className="t">{d.title}</span>
                <span>Saved</span>
              </NavLink>
            </li>
          ))}
          {library.items && library.items.length === 0 && <li className="none">No datasets yet.</li>}
          {library.loading && <li role="status">Loading datasets…</li>}
          {library.error && (
            <li role="alert">
              {library.error}
              <button onClick={() => void (library.items ? library.loadMore() : library.refresh())}>Retry</button>
            </li>
          )}
          {library.nextCursor && (
            <li>
              <button disabled={library.loading} onClick={() => void library.loadMore()}>
                Load more
              </button>
            </li>
          )}
        </ul>
      )}
    </nav>
  );
}
