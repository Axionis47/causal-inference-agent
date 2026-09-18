import { useEffect, useRef, useState } from "react";
import { composeAnswer } from "../compose";
import type { QuestionView, SessionView } from "../types";
import QuestionList from "./QuestionList";

export default function Composer({
  view,
  disabled,
  onSend,
  onEnd,
}: {
  view: SessionView;
  disabled: boolean;
  onSend: (text: string) => void;
  onEnd: () => void;
}) {
  const [free, setFree] = useState("");
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const ta = useRef<HTMLTextAreaElement>(null);
  const questions: QuestionView[] = view.questions;
  const before = view.phase === "before";

  // a new prompt clears the picked chips
  const promptKey = view.prompt?.text ?? "";
  useEffect(() => {
    setAnswers({});
  }, [promptKey]);

  const picked = Object.entries(answers).map(([i, a]) => ({ i: Number(i), question: questions[Number(i)]?.text ?? "", answer: a }));
  const text = composeAnswer(free, picked.map((p) => ({ question: p.question, answer: p.answer })));
  const canSend = !disabled && text.length > 0;

  const send = () => {
    if (!canSend) return;
    onSend(text);
    setFree("");
    setAnswers({});
  };

  return (
    <>
      <QuestionList
        questions={questions}
        answers={answers}
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
        {picked.length > 0 && (
          <div className="picked">
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
          placeholder={before ? "Answer in your own words, or pick the chips above and add what they miss." : "Ask what it found, why this design, what a flag means, or say what to change."}
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
              <button className="btn sm" disabled={disabled || !view.ready} onClick={() => onSend("run")} title={view.ready ? "Hand off to the analysis" : "Settle the open claims first"}>
                Run the analysis
              </button>
            )}
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
