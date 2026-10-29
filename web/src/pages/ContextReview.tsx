import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import type { components } from "../generated/schema";
import { api } from "../api";
import ContextStatements from "../components/ContextStatements";
import AddContextStatement from "../components/AddContextStatement";
import TopBar from "../components/TopBar";

type Review = components["schemas"]["ReviewView"];
type Edit = components["schemas"]["ContextEdit"];
type Disposition = components["schemas"]["ColumnDisposition"];

export default function ContextReview() {
  const { datasetId = "", reviewId = "" } = useParams();
  const navigate = useNavigate();
  const [review, setReview] = useState<Review | null>(null);
  const [section, setSection] = useState("dataset");
  const [nextStatements, setNextStatements] = useState<number | null>(null);
  const [statements, setStatements] = useState<components["schemas"]["KnowledgeStatement"][]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let active = true;
    api
      .review(datasetId, reviewId)
      .then((r) => {
        if (active) setReview(r);
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, [datasetId, reviewId]);
  useEffect(() => {
    let active = true;
    setStatements([]);
    api
      .reviewStatements(datasetId, reviewId, section)
      .then((r) => {
        if (active) {
          setStatements(r.statements);
          setNextStatements(r.next_offset);
        }
      })
      .catch((e) => {
        if (active) setError(String(e));
      });
    return () => {
      active = false;
    };
  }, [datasetId, reviewId, section, review?.generation]);
  async function save(edits: Edit[], dispositions: Disposition[]) {
    if (!review) return false;
    setBusy(true);
    setError("");
    try {
      setReview(await api.saveReview(datasetId, reviewId, { generation: review.generation, edits, dispositions }));
      return true;
    } catch (e) {
      setError(String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }
  async function edit(edits: Edit[]) {
    if (!review) return false;
    const targets = new Set(edits.map((e) => e.target_id));
    return save([...review.edits.filter((e) => !targets.has(e.target_id)), ...edits], review.dispositions);
  }
  async function disposition(status: "reviewed" | "unresolved") {
    if (!review) return false;
    await save(review.edits, [...review.dispositions.filter((d) => d.column_id !== section), { column_id: section, status, reason: "" }]);
  }
  const column = review?.columns.find((c) => c.column_id === section);
  const complete = review && review.columns.every((c) => review.dispositions.some((d) => d.column_id === c.column_id));
  return (
    <div className="wrap">
      <TopBar crumb={<Link to={`/datasets/${datasetId}`}>Dataset</Link>} />
      <h1>Review context before chat</h1>
      {error && (
        <div role="alert">
          <p>{error}</p>
          {review && (
            <button
              disabled={busy}
              onClick={() =>
                void api
                  .startReview(datasetId, review.question)
                  .then((r) => navigate(`/datasets/${datasetId}/reviews/${r.id}`))
                  .catch((e) => setError(String(e)))
              }
            >
              Start a review of the latest context
            </button>
          )}
          <p>Your previous review draft is retained. Changes are never merged automatically.</p>
        </div>
      )}
      {review ? (
        <>
          <p>
            <strong>Question:</strong> {review.question}
          </p>
          <p>
            Review the supplied meaning and source passages. Numerical profiling stays separate. Mark every column reviewed or unresolved. Column review alone
            does not confirm causal assumptions.
          </p>
          <nav aria-label="Context review sections">
            <button onClick={() => setSection("dataset")}>Dataset context</button>
            <button onClick={() => setSection("relationship")}>Relationships</button>
            {review.columns.map((c) => (
              <button key={c.column_id} onClick={() => setSection(c.column_id)} aria-pressed={section === c.column_id}>
                {c.name}: {review.dispositions.find((d) => d.column_id === c.column_id)?.status ?? "Not reviewed"}
              </button>
            ))}
          </nav>
          <h2>{column?.name ?? (section === "dataset" ? "Dataset context" : "Relationships")}</h2>
          <ContextStatements statements={statements} columns={review.columns} onEdit={edit} disabled={busy || !!review.completed_revision_id} />
          {nextStatements !== null && (
            <button
              onClick={() =>
                void api
                  .reviewStatements(datasetId, reviewId, section, nextStatements)
                  .then((v) => {
                    setStatements((old) => old.concat(v.statements));
                    setNextStatements(v.next_offset);
                  })
                  .catch((e) => setError(String(e)))
              }
            >
              Load more statements
            </button>
          )}
          <AddContextStatement key={section} section={section} columns={review.columns} save={edit} disabled={busy || !!review.completed_revision_id} />
          {column && (
            <div>
              <button disabled={busy} onClick={() => void disposition("reviewed")}>
                Mark this column reviewed
              </button>
              <button disabled={busy} onClick={() => void disposition("unresolved")}>
                Leave this column unresolved
              </button>
            </div>
          )}
          <p>
            {review.dispositions.length} of {review.columns.length} columns have a disposition.
          </p>
          <button
            disabled={busy}
            onClick={() =>
              void save(
                review.edits,
                review.columns.map(
                  (c) =>
                    review.dispositions.find((d) => d.column_id === c.column_id) ?? {
                      column_id: c.column_id,
                      status: "unresolved",
                      reason: "Left unresolved during review.",
                    },
                ),
              )
            }
          >
            Leave remaining columns unresolved
          </button>
          <button
            className="btn primary"
            disabled={busy || !complete}
            onClick={async () => {
              setBusy(true);
              setError("");
              try {
                const result = await api.createConversation(datasetId, review.id, review.generation);
                navigate(`/d/${result.id}`);
              } catch (e) {
                setError(String(e));
              } finally {
                setBusy(false);
              }
            }}
          >
            Confirm context and start chat
          </button>
        </>
      ) : (
        <p role="status">Loading review…</p>
      )}
    </div>
  );
}
