import { stageLabel } from "../stage";
import type { DatasetSummary } from "../types";

export default function StagePill({ d }: { d: DatasetSummary }) {
  const { text, tone } = stageLabel(d);
  return <span className={`pill${tone ? ` ${tone}` : ""}`}>{text}</span>;
}
