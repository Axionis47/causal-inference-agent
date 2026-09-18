import type { ReactNode } from "react";
import type { Turn } from "../types";

// Addresses in square brackets become small marks; numbered question lines get a heavier number.
const ADDR = /\[([a-z_]+:[^\]\s]+|decision\.[^\]\s]+|design\.[^\]\s]+|primary\.[^\]\s]+|feasibility\.[^\]\s]+|run\.[^\]\s]+|dataset|profile|unstated)\]/g;

export function richText(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const lines = text.split("\n");
  lines.forEach((line, li) => {
    const m = /^(\d+)\.\s/.exec(line);
    let rest = line;
    if (m) {
      out.push(
        <span className="num" key={`n${li}`}>
          {m[1]}.{" "}
        </span>,
      );
      rest = line.slice(m[0].length);
    }
    let last = 0;
    let k = 0;
    for (const a of rest.matchAll(ADDR)) {
      if (a.index! > last) out.push(rest.slice(last, a.index));
      out.push(
        <span className="addr" key={`a${li}-${k++}`} title="an address in the run's artifacts">
          {a[1]}
        </span>,
      );
      last = a.index! + a[0].length;
    }
    if (last < rest.length) out.push(rest.slice(last));
    if (li < lines.length - 1) out.push("\n");
  });
  return out;
}

function when(at: string): string {
  try {
    return new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

export default function Message({ t }: { t: Turn }) {
  if (t.role === "system") {
    return <div className={`turn system${t.kind === "error" ? " error" : ""}`}>{t.text}</div>;
  }
  return (
    <div className={`turn ${t.role}`}>
      <div className="who">
        <span>{t.role === "user" ? "you" : "desk"}</span>
        <span>{t.phase === "after" ? "after the run" : "before the run"}</span>
        <span>{when(t.at)}</span>
      </div>
      <div className="body">{t.role === "assistant" ? richText(t.text) : t.text}</div>
    </div>
  );
}
