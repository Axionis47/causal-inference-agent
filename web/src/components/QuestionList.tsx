import { useState } from "react";
import type { AskItemView, QuestionView } from "../types";

// The desk's question as chips (docs/design.md, "A question with its recommended option"): the recommended option first and
// marked, the others after, "other" frees the text box. A readback is a questionnaire: each draft is one line with the
// desk's reading recommended, the other legal answers as chips, and "other" for a value of the person's own; a changed line
// is sent back as `address = value`.
function chipsFor(q: QuestionView): string[] {
  const base = q.kind === "choose" ? q.options : q.kind === "confirm" ? (q.options.length ? q.options : ["Yes, that's right", "No"]) : [];
  if (!q.recommended || !base.includes(q.recommended)) return base;
  return [q.recommended, ...base.filter((c) => c !== q.recommended)];
}

function ItemRow({ item, value, onEdit }: { item: AskItemView; value: string | undefined; onEdit: (address: string, value: string | null) => void }) {
  const [other, setOther] = useState(false);
  const [draft, setDraft] = useState("");
  const chips = item.recommended ? [item.recommended, ...item.options.filter((o) => o !== item.recommended)] : item.options;
  return (
    <div className="qrow item">
      <span className="t">
        <span className="addr" title={item.address}>
          {item.label}
        </span>
        {!item.options.length && <span className="muted"> {item.value}</span>}
      </span>
      {chips.map((c) => {
        const rec = c === item.recommended;
        const on = value === c || (value === undefined && rec && !other);
        return (
          <button
            key={c}
            type="button"
            className={`chip${on ? " on" : ""}${rec ? " rec" : ""}`}
            title={rec ? "what the desk read; the line stands unless you change it" : undefined}
            onClick={() => {
              setOther(false);
              onEdit(item.address, rec ? null : c);
            }}
          >
            {c}
            {rec && <span className="recmark"> · as read</span>}
          </button>
        );
      })}
      {other ? (
        <input
          className="other"
          autoFocus
          value={draft}
          placeholder="your own value"
          aria-label={`${item.address} other`}
          onChange={(e) => {
            setDraft(e.target.value);
            onEdit(item.address, e.target.value.trim() || null);
          }}
        />
      ) : (
        <button
          type="button"
          className="chip"
          onClick={() => {
            setOther(true);
            onEdit(item.address, null);
          }}
        >
          other
        </button>
      )}
    </div>
  );
}

export default function QuestionList({
  questions,
  answers,
  edits,
  onPick,
  onEdit,
  onOther,
}: {
  questions: QuestionView[];
  answers: Record<number, string>;
  edits: Record<string, string>;
  onPick: (i: number, answer: string | null) => void;
  onEdit: (address: string, value: string | null) => void;
  onOther: () => void;
}) {
  if (!questions.length) return null;
  return (
    <div className="qs" aria-label="Open questions">
      {questions.map((q, i) => (
        <div key={i}>
          <div className="qrow">
            <span className="n">{i + 1}.</span>
            <span className="t">
              {q.items.length ? "Each line below is what I read; change any, or say it is all right." : q.text}
              {q.because.length > 0 && <span className="why"> · asked for {q.because.join(", ")}</span>}
            </span>
            {chipsFor(q).map((c) => {
              const on = answers[i] === c;
              const rec = c === q.recommended;
              return (
                <button key={c} className={`chip${on ? " on" : ""}${rec ? " rec" : ""}`} onClick={() => onPick(i, on ? null : c)} type="button">
                  {c}
                  {rec && <span className="recmark"> · recommended</span>}
                </button>
              );
            })}
            {q.kind !== "confirm" && (
              <button className="chip" type="button" onClick={onOther} title="Answer in your own words below">
                other
              </button>
            )}
          </div>
          {q.items.map((item) => (
            <ItemRow key={item.address} item={item} value={edits[item.address]} onEdit={onEdit} />
          ))}
        </div>
      ))}
    </div>
  );
}
