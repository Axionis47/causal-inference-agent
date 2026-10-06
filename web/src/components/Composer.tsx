import { useEffect, useRef, useState } from "react";
import { composeAnswer } from "../compose";
import type { QuestionView, SessionView } from "../types";
import HandoffPanel from "./HandoffPanel";
import QuestionList from "./QuestionList";

export default function Composer({
  view,
  disabled,
  onSend,
  onEnd,
  onNew,
}: {
  view: SessionView;
  disabled: boolean;
  onSend: (text: string) => void;
  onEnd: () => void;
  onNew: () => void;
}) {
  const [free, setFree] = useState("");
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [edits, setEdits] = useState<Record<string, string>>({});
  const ta = useRef<HTMLTextAreaElement>(null);
  const questions: QuestionView[] = view.questions;
  const before = view.phase === "before";

  // a new prompt clears the picked chips
  const promptKey = view.prompt?.text ?? "";
  useEffect(() => {
    setAnswers({});
    setEdits({});
  }, [promptKey]);

  const picked = Object.entries(answers).map(([i, a]) => ({ i: Number(i), question: questions[Number(i)]?.text ?? "", answer: a }));
  const text = composeAnswer(
    free,
    picked.map((p) => ({ question: p.question, answer: p.answer })),
    Object.entries(edits).map(([address, value]) => ({ address, value })),
  );
  const canSend = !disabled && text.length > 0;

  const send = () => {
    if (!canSend) return;
    onSend(text);
    setFree("");
    setAnswers({});
    setEdits({});
  };

  return (
    <>
      {before && view.prompt?.kind === "ready" && view.handoff && (
        <HandoffPanel h={view.handoff} disabled={disabled} onEdit={(address, value) => onSend(`${address} = ${value}`)} />
      )}
      <QuestionList
        questions={questions}
        answers={answers}
        edits={edits}
        onEdit={(address, value) =>
          setEdits((prev) => {
            const next = { ...prev };
            if (value === null) delete next[address];
            else next[address] = value;
            return next;
          })
        }
        onOther={() => ta.current?.focus()}
        onPick={(i, a) => {
          setAnswers((prev) => {
            const next = { ...prev };
            if (a === null) delete next[i];
            else next[i] = a;
            return next;
          });
          if (a === "No") ta.current?.focus();
        }}
      />
      <div className="composer">
        {(picked.length > 0 || Object.keys(edits).length > 0) && (
          <div className="picked">
            {Object.entries(edits).map(([address, value]) => (
              <span className="chip on" key={address}>
                {address} = {value}
                <button
                  aria-label="remove"
                  onClick={() =>
                    setEdits((prev) => {
                      const next = { ...prev };
                      delete next[address];
                      return next;
                    })
                  }
                >
                  ×
                </button>
              </span>
            ))}
            {picked.map((p) => (
              <span className="chip on" key={p.i}>
                {p.i + 1}. {p.answer}
                <button
                  aria-label="remove"
                  onClick={() =>
                    setAnswers((prev) => {
                      const next = { ...prev };
                      delete next[p.i];
                      return next;
                    })
                  }
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        <textarea
          ref={ta}
          value={free}
          disabled={disabled}
          placeholder={
            !before
              ? "Ask what it found, why this design, what a flag means, or say what to change."
              : view.prompt?.kind === "question"
                ? "What changed, and what might it have affected? Name them as they appear in the columns."
                : view.prompt?.kind === "ask"
                  ? "Answer in your own words, or pick a chip above and add what it misses."
                  : view.prompt?.kind === "ready"
                    ? "Say yes to hand off, or correct any line of the design."
                    : "Tell me about the data, ask about it, or ask for a picture."
          }
          onChange={(e) => setFree(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) send();
          }}
        />
        <div className="bar">
          <div className="left">
            <button className="btn primary sm" disabled={!canSend} onClick={send}>
              Send
            </button>
            {before && (
              <button
                className="btn sm"
                disabled={disabled}
                onClick={() => onSend(view.prompt?.kind === "ready" ? "yes, run it" : "run the analysis")}
                title={view.prompt?.kind === "ready" ? "Hand off to the analysis" : "The desk will ask what it still needs, then show the design"}
              >
                {view.prompt?.kind === "ready" ? "Yes, run it" : "Run the analysis"}
              </button>
            )}
            <button className="btn quiet sm" disabled={disabled} onClick={onNew} title="A new question of the same file; what is known carries over">
              New question
            </button>
            <button className="btn quiet sm" disabled={disabled} onClick={onEnd}>
              End conversation
            </button>
          </div>
          <span className="hint">⌘↵ sends</span>
        </div>
      </div>
    </>
  );
}
