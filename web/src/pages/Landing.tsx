import { Link } from "react-router-dom";
import DatasetCard from "../components/DatasetCard";
import { useDatasets } from "../components/Shell";
import TopBar from "../components/TopBar";

export default function Landing() {
  const { items, error, loading, nextCursor, refresh, loadMore } = useDatasets();
  return (
    <div className="wrap">
      <TopBar
        crumb="Datasets"
        right={
          <Link className="btn primary" to="/new">
            New dataset
          </Link>
        }
      />
      <section className="hero">
        <div>
          <h1>Your datasets</h1>
          <p className="lede">Return to your saved CSVs and original study context.</p>
        </div>
      </section>
      <button className="btn" disabled={loading} onClick={() => void refresh()}>
        Refresh datasets
      </button>
      {loading && <p role="status">Loading datasets…</p>}
      {error && (
        <div role="alert">
          <p>{error}</p>
          <button onClick={() => void (items ? loadMore() : refresh())}>Retry</button>
        </div>
      )}
      {items?.length === 0 && <p>No datasets yet. Upload a CSV to get started.</p>}
      <div className="grid">
        {items?.map((d) => (
          <DatasetCard key={d.dataset_id} d={d} />
        ))}
      </div>
      {nextCursor && (
        <button className="btn" disabled={loading} onClick={() => void loadMore()}>
          Load more
        </button>
      )}
    </div>
  );
}
