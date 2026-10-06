// The one string the graph receives for a turn. Chips answer questions; free text carries the rest; a line of the
// readback changed on the page becomes `address = value`, the person's word on that field, which no model reads when the
// message holds nothing else. The graph reads prose, so each answered question is quoted beside its answer rather than numbered.

export interface Answered {
  question: string;
  answer: string;
}

export interface Edit {
  address: string;
  value: string;
}

export function composeAnswer(free: string, answered: Answered[], edits: Edit[] = []): string {
  const parts: string[] = [];
  for (const a of answered) {
    const q = a.question.trim().replace(/\s+/g, " ");
    const ans = a.answer.trim();
    if (!ans) continue;
    parts.push(`On "${q}": ${ans}.`);
  }
  const rest = free.trim();
  if (rest) parts.push(rest);
  for (const e of edits) {
    const v = e.value.trim();
    if (v) parts.push(`${e.address} = ${v}`);
  }
  return parts.join("\n");
}
