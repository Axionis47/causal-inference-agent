import { useState } from "react";
import type { components } from "../generated/schema";

type Evidence = components["schemas"]["EvidenceView"];
export default function CitedText({ text, chat }: { text: string; chat: string }) {
  const [evidence, setEvidence] = useState<Evidence | null>(null),
    [error, setError] = useState("");
  async function show(address: string) {
    try {
      const response = await fetch(`/api/conversations/${encodeURIComponent(chat)}/evidence?address=${encodeURIComponent(address)}`);
      if (!response.ok) throw new Error("This source is unavailable in the selected chat.");
      setEvidence(await response.json());
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  return (
    <>
      <p style={{ whiteSpace: "pre-wrap" }}>
        {text.split(/(\[(?:context:|artifact:|operation:|chat\.question)[^\]]*\])/g).map((part, i) =>
          /^\[(?:context:|artifact:|operation:|chat\.question)/.test(part) ? (
            <button key={i} className="citation" onClick={() => void show(part.slice(1, -1))}>
              Source
            </button>
          ) : (
            part
          ),
        )}
      </p>
      {error && <p role="alert">{error}</p>}
      {evidence && (
        <aside aria-label="Cited evidence">
          <strong>{evidence.title}</strong>
          <p>{evidence.text}</p>
          {evidence.quotes?.map((q, i) => (
            <blockquote key={i}>{q}</blockquote>
          ))}
          <button onClick={() => setEvidence(null)}>Close source</button>
        </aside>
      )}
    </>
  );
}
