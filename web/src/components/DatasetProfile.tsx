import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ProfileCheck, ProfileIssue, ProfileView, SavedColumn, SavedProfile } from "../types";

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

function Findings({ checks, issues }: { checks: ProfileCheck[]; issues: ProfileIssue[] }) {
  return (
    <>
      <ul aria-label="Check coverage">
        {checks.map((c) => (
          <li key={c.name}>
            {c.name}: {c.status}
            {c.reason && `: ${c.reason}`}
          </li>
        ))}
      </ul>
      {issues.map((i) => (
        <div key={i.id} className="profile-issue">
          <strong>{i.reason}</strong>
          {i.affected_count !== null && <p>Affected values: {i.affected_count}</p>}
          {i.examples.length > 0 && (
            <ul>
              {i.examples.map((e, n) => (
                <li key={n}>
                  <code style={{ whiteSpace: "pre-wrap" }}>{e}</code>
                </li>
              ))}
            </ul>
          )}
          {i.examples_truncated && <p>Examples are limited or shortened.</p>}
        </div>
      ))}
    </>
  );
}

function ColumnDetails({ datasetId, profile, columnId }: { datasetId: string; profile: SavedProfile; columnId: string }) {
  const [result, setResult] = useState<SavedColumn | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    api
      .profileColumn(datasetId, profile.profile_id, columnId)
      .then((value) => {
        if (active) setResult(value);
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, [datasetId, profile.profile_id, columnId, attempt]);
  if (error)
    return (
      <div role="alert">
        {error}
        <button
          onClick={() => {
            setError(null);
            setAttempt((a) => a + 1);
          }}
        >
          Retry column
        </button>
      </div>
    );
  if (!result) return <p role="status">Loading column…</p>;
  const col = result.result;
  return (
    <div>
      <p>
        Status: {col.status}. Suggested type: {col.suggested_dtype}. No values were changed.
      </p>
      <dl className="profile-facts">
        <dt>Empty fields</dt>
        <dd>{col.empty_count ?? "Unavailable"}</dd>
        <dt>Whitespace-only fields</dt>
        <dd>{col.whitespace_only_count ?? "Unavailable"}</dd>
        <dt>Distinct non-empty values</dt>
        <dd>{col.distinct_count ?? "Unavailable"}</dd>
        <dt>Numeric parsing successes</dt>
        <dd>{col.numeric_parse_count ?? "Unavailable"}</dd>
        <dt>Numeric parsing exceptions</dt>
        <dd>{col.numeric_exception_count ?? "Unavailable"}</dd>
      </dl>
      {col.numeric && (
        <>
          <h4>Numerical summary</h4>
          <dl className="profile-facts">
            {Object.entries(col.numeric).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
      {col.top_values.length > 0 && (
        <>
          <h4>Most frequent non-empty values</h4>
          <table>
            <thead>
              <tr>
                <th>Original value</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {col.top_values.map((v, i) => (
                <tr key={i}>
                  <td>
                    <code style={{ whiteSpace: "pre-wrap" }}>{v.value}</code>
                    {v.truncated && " (shortened)"}
                  </td>
                  <td>{v.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {col.suspected_sentinels.length > 0 && (
        <>
          <h4>Suspected sentinel values</h4>
          <p>These remain original values, not confirmed missing data.</p>
          <ul>
            {col.suspected_sentinels.map((v, i) => (
              <li key={i}>
                <code style={{ whiteSpace: "pre-wrap" }}>{v.value}</code>: {v.count}
                {v.truncated && " (value shortened)"}
              </li>
            ))}
          </ul>
          {col.sentinel_values_truncated && <p>Only the eight most frequent candidates are shown.</p>}
        </>
      )}
      <Findings checks={col.checks} issues={col.issues} />
    </div>
  );
}

export default function DatasetProfile({ datasetId }: { datasetId: string }) {
  const [view, setView] = useState<ProfileView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opened, setOpened] = useState<string | null>(null);
  const generation = useRef(0);
  const inFlight = useRef(false);
  useEffect(() => {
    const ticket = ++generation.current;
    api
      .profile(datasetId)
      .then((value) => {
        if (ticket === generation.current) setView(value);
      })
      .catch((e) => {
        if (ticket === generation.current) setError(message(e));
      });
    return () => {
      generation.current++;
    };
  }, [datasetId]);
  async function request(generate: boolean) {
    if (inFlight.current) return;
    inFlight.current = true;
    const ticket = ++generation.current;
    setBusy(true);
    setError(null);
    try {
      const value = await (generate ? api.generateProfile(datasetId) : api.profile(datasetId));
      if (ticket === generation.current) {
        setView(value);
        setOpened(null);
      }
    } catch (e) {
      if (ticket === generation.current) setError(message(e));
    } finally {
      inFlight.current = false;
      if (ticket === generation.current) setBusy(false);
    }
  }
  const profile = view?.profile;
  return (
    <section className="dataset-profile" aria-label="Dataset profile">
      <h2>Dataset profile</h2>
      <p>Read-only measurements of the original CSV. Suggested types, sentinel values and candidate keys are not confirmed meanings.</p>
      {!view && !error && <p role="status">Loading saved profile…</p>}
      {error && (
        <div role="alert">
          {error}{" "}
          <button disabled={busy} onClick={() => void request(false)}>
            Reload profile
          </button>
        </div>
      )}
      {view?.status === "not_profiled" && <p>No profile yet.</p>}
      {view && view.status !== "complete" && (
        <button disabled={busy} onClick={() => void request(true)}>
          {busy ? "Profiling…" : profile ? "Retry profiling" : "Profile dataset"}
        </button>
      )}
      {busy && <p role="status">Profiling the original CSV…</p>}
      {profile && (
        <>
          <p>
            Profile status: <strong>{profile.status}</strong>
          </p>
          <p>
            Source version: <code>{profile.version_id}</code>
            <br />
            Profiled {new Date(profile.updated_at).toLocaleString()}
          </p>
          <dl className="profile-facts">
            <dt>Rows</dt>
            <dd>{profile.dataset.rows ?? "Unavailable"}</dd>
            <dt>Columns</dt>
            <dd>{profile.dataset.columns ?? "Unavailable"}</dd>
            <dt>Duplicate rows</dt>
            <dd>{profile.dataset.duplicate_rows ?? "Unavailable"}</dd>
          </dl>
          <h3>Candidate keys</h3>
          <p>
            {profile.dataset.candidate_keys.length
              ? profile.dataset.candidate_keys
                  .map((key) => key.map((pos) => profile.columns.find((c) => c.position === pos)?.name ?? String(pos)).join(" + "))
                  .join("; ")
              : "No candidate found within the search limits."}
          </p>
          <p>{profile.dataset.key_search_limit} A key does not establish what a row represents.</p>
          <Findings checks={profile.dataset.checks} issues={profile.dataset.issues} />
          <h3>Columns</h3>
          {profile.columns.map((col) => (
            <div className="profile-column" key={col.column_id}>
              <button aria-expanded={opened === col.column_id} onClick={() => setOpened(opened === col.column_id ? null : col.column_id)}>
                {col.name}
              </button>
              <span>
                {" "}
                Suggested type: {col.suggested_dtype} · {col.status} · {col.issue_count} issues
              </span>
              {opened === col.column_id && (
                <ColumnDetails
                  key={`${profile.profile_id}:${profile.updated_at}:${col.column_id}`}
                  datasetId={datasetId}
                  profile={profile}
                  columnId={col.column_id}
                />
              )}
            </div>
          ))}
        </>
      )}
    </section>
  );
}
