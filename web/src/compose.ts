// The one string the graph receives for a turn. Chips answer questions; free text carries the rest.
// The graph reads prose, so each answered question is quoted beside its answer rather than numbered.

export interface Answered {
  question: string;
  answer: string;
}

export function composeAnswer(free: string, answered: Answered[]): string {
  const parts: string[] = [];
  for (const a of answered) {
    const q = a.question.trim().replace(/\s+/g, " ");
    const ans = a.answer.trim();
    if (!ans) continue;
    parts.push(`On "${q}": ${ans}.`);
  }
  const rest = free.trim();
  if (rest) parts.push(rest);
  return parts.join("\n");
}
