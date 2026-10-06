const WORDS: Record<string, string> = {
  starting: "starting",
  load: "opening the memory of this file",
  ask_question: "waiting for the question",
  mine: "reading the note",
  prefilter: "sorting the columns",
  read_question: "reading the question",
  check: "checking what is known against the file",
  probe_fit: "probing the file for each design",
  ask: "writing the reply",
  convince: "assembling the design",
  listen: "waiting",
  infer: "reading what you said",
  draw: "drawing it",
  explain: "answering",
  fit: "fitting the families",
  decide: "choosing the design",
  gate: "checking the decision",
  design: "writing the design brief",
  handoff: "writing the hand-off",
  run: "running the analysis; this takes a few minutes",
  ask_back: "bringing back what the analysis asked",
  brief: "writing the brief",
  talk: "waiting",
  turn: "thinking about your message",
  answer: "answering",
  revise: "applying the change to what is known",
  what_if: "trying it on a copy",
  requestion: "taking the new question",
  draw_after: "drawing it",
};

export default function Activity({ node }: { node: string }) {
  const text = WORDS[node] ?? node.replace(/_/g, " ");
  return (
    <div className="activity" role="status">
      <span className="spin" aria-hidden="true" />
      <span>{text}…</span>
    </div>
  );
}
