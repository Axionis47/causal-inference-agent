const WORDS: Record<string, string> = {
  starting: "starting",
  load: "reading the description",
  extract: "reading what you said",
  check: "checking it against the file",
  probe: "probing the file for each design",
  status: "updating the table",
  respond: "writing the next questions",
  listen: "waiting",
  write_pack: "writing the dataset for the analysis",
  run: "running the analysis; this takes a few minutes",
  brief: "writing the brief",
  talk: "waiting",
  turn: "thinking about your message",
  answer: "answering",
  revise: "applying the change to the claims",
  requestion: "taking the new question",
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
