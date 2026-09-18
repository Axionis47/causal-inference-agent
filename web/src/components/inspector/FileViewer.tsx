import { useEffect, useState } from "react";
import { api } from "../../api";
import JsonView from "../JsonView";

const JSON_FILES = new Set(["artifacts.json", "design.json"]);
const TEXT = new Set(["report.md", "design.md"]);

export const viewable = (name: string) => JSON_FILES.has(name) || TEXT.has(name);

/** One run file, shown whole. Depends on the two strings only, so polling never refetches it. */
export default function FileViewer({ runId, name }: { runId: string; name: string }) {
  const [text, setText] = useState<string | null>(null);
  const [data, setData] = useState<unknown>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let on = true;
    setText(null);
    setData(null);
    setErr(null);
    api
      .fileText(runId, name)
      .then((t) => {
        if (!on) return;
        if (JSON_FILES.has(name)) setData(JSON.parse(t));
        else setText(t);
      })
      .catch((e) => on && setErr(String(e)));
    return () => {
      on = false;
    };
  }, [runId, name]);
  if (err) return <p className="err">{err}</p>;
  if (data !== null)
    return (
      <div className="viewer fill">
        <JsonView value={data} />
      </div>
    );
  if (text !== null)
    return (
      <div className="viewer fill">
        <pre>{text}</pre>
      </div>
    );
  return <p className="muted">Loading…</p>;
}
