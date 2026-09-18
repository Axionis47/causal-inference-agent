import type { QuestionView } from "../types";

export default function QuestionList({
  questions,
  answers,
  onPick,
}: {
  questions: QuestionView[];
  answers: Record<number, string>;
  onPick: (i: number, answer: string | null) => void;
}) {
  if (!questions.length) return null;
  return (
    <div className="qs" aria-label="Open questions">
      {questions.map((q, i) => {
        const chips = q.kind === "choose" ? q.options : q.kind === "confirm" ? ["Yes, that's right", "No"] : [];
        return (
          <div className="qrow" key={i}>
            <span className="n">{i + 1}.</span>
            <span className="t">{q.text}</span>
            {chips.map((c) => {
              const on = answers[i] === c;
              return (
                <button key={c} className={`chip${on ? " on" : ""}`} onClick={() => onPick(i, on ? null : c)} type="button">
                  {c}
                </button>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
