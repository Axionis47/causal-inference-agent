import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api";
import ConfirmDialog from "../components/ConfirmDialog";
import DatasetCard from "../components/DatasetCard";
import { useDatasets } from "../components/Shell";
import TopBar from "../components/TopBar";
import type { DatasetSummary } from "../types";

export default function Landing() {
  const { items, error: listError, refresh } = useDatasets();
  const [error, setError] = useState<string | null>(null);
  const [victim, setVictim] = useState<DatasetSummary | null>(null);
  const [busy, setBusy] = useState(false);

  const remove = async () => {
    if (!victim) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteDataset(victim.name);
      setVictim(null);
      await refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const newButton = (
    <Link className="btn primary" to="/new">
      New dataset
    </Link>
  );

  return (
    <div className="wrap">
      <TopBar crumb="Datasets" right={newButton} />
      <section className="hero">
        <div>
          <h1>Datasets</h1>
          <p className="lede">Each one is a table, a question, and a conversation that settles what the table can answer.</p>
        </div>
      </section>
      {(error ?? listError) && <p className="err">{error ?? listError}</p>}
      {items && items.length === 0 && (
        <div className="empty">
          <p>No datasets yet. Add one to start.</p>
          {newButton}
        </div>
      )}
      <div className="grid">{items?.map((d) => <DatasetCard key={d.name} d={d} onDelete={setVictim} />)}</div>
      <ConfirmDialog
        open={victim !== null}
        title={`Delete ${victim?.title ?? ""}?`}
        body={`This removes the dataset "${victim?.name}", its description, its claims, the conversation, and every run it produced. The uploaded file goes too unless another dataset shares it.`}
        confirmLabel="Delete dataset"
        danger
        busy={busy}
        onConfirm={remove}
        onCancel={() => !busy && setVictim(null)}
      />
    </div>
  );
}
