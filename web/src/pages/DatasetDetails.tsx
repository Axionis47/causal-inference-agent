import { useEffect, useState } from "react";
import { Link, useLocation, useParams, useNavigate } from "react-router-dom";
import { api } from "../api";
import type { IngestedDataset } from "../types";
import DatasetProfile from "../components/DatasetProfile";
import DatasetContext from "../components/DatasetContext";
import TopBar from "../components/TopBar";

export default function DatasetDetails() {
  const { datasetId = "" } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [chats, setChats] = useState<{ id: string; title: string }[]>([]);
  const [chatError, setChatError] = useState("");
  const [starting, setStarting] = useState(false);
  const [question, setQuestion] = useState("");
  useEffect(() => {
    let active = true;
    api
      .conversations(datasetId)
      .then((data) => {
        if (active) setChats(data);
      })
      .catch((e) => {
        if (active) setChatError(String(e));
      });
    return () => {
      active = false;
    };
  }, [datasetId]);
  const [result, setResult] = useState<{ id: string; data?: IngestedDataset; error?: string } | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    api
      .dataset(datasetId)
      .then((data) => {
        if (active) setResult({ id: datasetId, data });
      })
      .catch((e) => {
        if (active) setResult({ id: datasetId, error: e instanceof Error ? e.message : String(e) });
      });
    return () => {
      active = false;
    };
  }, [datasetId, attempt]);
  const current = result?.id === datasetId ? result : null;
  return (
    <div className="wrap">
      <TopBar
        crumb={
          <>
            <Link to="/">Datasets</Link> / details
          </>
        }
      />
      {!current && <p role="status">Loading dataset…</p>}
      {current?.error && (
        <div role="alert">
          <p>{current.error}</p>
          <button
            onClick={() => {
              setResult(null);
              setAttempt((x) => x + 1);
            }}
          >
            Retry
          </button>
        </div>
      )}
      {current?.data && (
        <section className="stage">
          {location.state?.saved && <p role="status">Dataset saved</p>}
          <h1>{current.data.title}</h1>
          <p>
            {current.data.filename} · {current.data.size_bytes.toLocaleString()} bytes · Saved
          </p>
          <p>Uploaded {new Date(current.data.created_at).toLocaleString()}</p>
          <h2>Original study context</h2>
          <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{current.data.initial_context ?? "Not supplied"}</p>
          <a className="btn" href={`/api/datasets/${datasetId}/file`}>
            Download original CSV
          </a>
          <section>
            <h2>Conversations</h2>
            {chatError && <p role="alert">{chatError}</p>}
            <label>
              What question would you like to explore?
              <textarea value={question} onChange={(e) => setQuestion(e.target.value)} />
            </label>
            <button
              disabled={starting || !question.trim()}
              onClick={async () => {
                setStarting(true);
                setChatError("");
                try {
                  const review = await api.startReview(datasetId, question);
                  navigate(`/datasets/${datasetId}/reviews/${review.id}`);
                } catch (e) {
                  setChatError(String(e));
                } finally {
                  setStarting(false);
                }
              }}
            >
              {starting ? "Opening…" : "Review context for this question"}
            </button>
            {chatError && (
              <button
                disabled={starting}
                onClick={async () => {
                  setStarting(true);
                  setChatError("");
                  try {
                    await api.generateProfile(datasetId);
                    await api.prepareContext(datasetId);
                  } catch (e) {
                    setChatError(String(e));
                  } finally {
                    setStarting(false);
                  }
                }}
              >
                Prepare profile and mapped context
              </button>
            )}
            <ul>
              {chats.map((chat) => (
                <li key={chat.id}>
                  <Link to={`/d/${chat.id}`}>{chat.title}</Link>
                </li>
              ))}
            </ul>
          </section>
          <DatasetProfile key={datasetId} datasetId={datasetId} />
          <DatasetContext key={`context:${datasetId}`} datasetId={datasetId} />
        </section>
      )}
    </div>
  );
}
