import {
  ChatWorkspace,
  type ConversationPanelState,
} from "@/features/chat/workspace";
import { AnalysisPanel } from "@/features/analysis/panel";

const analysisPanel = (state: ConversationPanelState) => (
  <AnalysisPanel {...state} />
);

export function ConversationPage(props: {
  datasetId: string;
  versionId: string;
  conversation?: string;
  revision?: string;
  analysis?: string;
  view?: "chat" | "analysis";
}) {
  const view = props.view ?? (props.analysis ? "analysis" : "chat");
  return (
    <ChatWorkspace {...props} view={view} renderSidePanel={analysisPanel} />
  );
}
