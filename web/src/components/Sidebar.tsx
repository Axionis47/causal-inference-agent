import { Link, NavLink } from "react-router-dom";
import type { DatasetSummary } from "../types";
import StagePill from "./StagePill";

export default function Sidebar({
  items,
  current,
  open,
  onToggle,
  onNav,
}: {
  items: DatasetSummary[] | null;
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
          {items?.map((d) => (
            <li key={d.name}>
              <NavLink to={`/d/${encodeURIComponent(d.name)}`} className={d.name === current ? "on" : undefined} aria-current={d.name === current ? "page" : undefined} onClick={onNav}>
                <span className="t">{d.title}</span>
                <StagePill d={d} />
              </NavLink>
            </li>
          ))}
          {items && items.length === 0 && <li className="none">No datasets yet.</li>}
        </ul>
      )}
    </nav>
  );
}
