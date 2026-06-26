import { Link, useLocation, useSearch } from "@tanstack/react-router";
import { api, type Dataset } from "@/lib/api/client";

export function WorkspaceNavigation({ dataset }: { dataset: Dataset }) {
  const pathname = useLocation({ select: (location) => location.pathname });
  const search = useSearch({ strict: false });
  const conversationId = search.conversation ?? search.fromConversation;
  const conversation = api.useQuery(
    "get",
    "/api/conversations/{conversation_id}",
    { params: { path: { conversation_id: conversationId ?? "" } } },
    { enabled: Boolean(conversationId) },
  );
  const saved =
    conversation.data?.version_id === dataset.version.id
      ? conversation.data
      : undefined;
  const context = pathname.includes("/context/");
  const chat = pathname.includes("/chat/");
  const analysis =
    chat &&
    (search.view === "analysis" || (!search.view && Boolean(search.analysis)));
  const params = { datasetId: dataset.id, versionId: dataset.version.id };
  return (
    <nav className="workspace-navigation" aria-label="Dataset workspace">
      <Link
        to="/datasets/$datasetId"
        params={params}
        search={{ page: 1, fromConversation: saved?.id }}
        aria-current={!context && !chat ? "page" : undefined}
      >
        Data
      </Link>
      <Link
        to="/datasets/$datasetId/context/$versionId"
        params={params}
        search={{
          revision: search.revision ?? saved?.context_revision_id ?? undefined,
          section: "dataset",
          fromConversation: saved?.id,
        }}
        aria-current={context ? "page" : undefined}
      >
        Context
      </Link>
      <Link
        to="/datasets/$datasetId/chat/$versionId"
        params={params}
        search={{
          conversation: saved?.id,
          analysis: search.analysis,
          view: "chat",
        }}
        resetScroll={false}
        aria-current={chat && !analysis ? "page" : undefined}
      >
        Chat
      </Link>
      <Link
        to="/datasets/$datasetId/chat/$versionId"
        params={params}
        search={{
          conversation: saved?.id,
          analysis: search.analysis,
          view: "analysis",
        }}
        resetScroll={false}
        aria-current={analysis ? "page" : undefined}
      >
        Analysis
      </Link>
    </nav>
  );
}
