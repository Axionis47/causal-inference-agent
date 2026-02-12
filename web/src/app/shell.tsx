import { useState } from "react";
import {
  Link,
  Outlet,
  useLocation,
  useNavigate,
  useParams,
  useSearch,
} from "@tanstack/react-router";
import { ChevronDown, Database, LogOut } from "lucide-react";
import { api, queryClient } from "@/lib/api/client";
import { Brand } from "@/components/brand";
import { Button } from "@/components/ui/button";
import { ErrorMessage } from "@/components/feedback";
import { DatasetNavigation } from "@/features/datasets/navigation";
import { WorkspaceNavigation } from "./workspace-navigation";

export function Shell() {
  const user = api.useQuery("get", "/api/auth/me");
  const { datasetId } = useParams({ strict: false });
  const pathname = useLocation({ select: (location) => location.pathname });
  const search = useSearch({ strict: false });
  const dataset = api.useQuery(
    "get",
    "/api/datasets/{dataset_id}",
    { params: { path: { dataset_id: datasetId ?? "" } } },
    { enabled: !!datasetId },
  );
  const [datasetsOpen, setDatasetsOpen] = useState(false);
  const surface = pathname.includes("/context/")
    ? "Context"
    : pathname.includes("/chat/")
      ? search.view === "analysis" || (!search.view && search.analysis)
        ? "Analysis"
        : "Chat"
      : "Preview";
  const navigate = useNavigate();
  const logout = api.useMutation("post", "/api/auth/logout", {
    onSuccess: async () => {
      queryClient.clear();
      await navigate({ to: "/login" });
    },
  });
  return (
    <div className="workspace">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <aside className="sidebar" aria-label="Dataset navigation">
        <div className="sidebar-top">
          <Brand />
          <Button
            className="dataset-toggle"
            variant="outline"
            size="sm"
            aria-expanded={datasetsOpen}
            aria-controls="dataset-navigation"
            onClick={() => setDatasetsOpen((open) => !open)}
          >
            <Database size={15} aria-hidden="true" />
            Datasets
            <ChevronDown size={14} aria-hidden="true" />
          </Button>
        </div>
        <div
          className="sidebar-body"
          id="dataset-navigation"
          data-open={datasetsOpen}
        >
          <DatasetNavigation
            selectedId={datasetId}
            current={dataset.data}
            onNavigate={() => setDatasetsOpen(false)}
          />
        </div>
        <div className="sidebar-bottom">
          <div>
            <span className="status-dot" />
            {user.data ? "Session active" : "Checking session"}
          </div>
          <div>CAUSAL WORKSPACE / 0.1</div>
        </div>
      </aside>
      <div className="min-w-0">
        <div className="workspace-header">
          <header className="topbar">
            {datasetId ? (
              <div className="workspace-dataset" aria-label="Current dataset">
                <span className="workspace-dataset-tag">
                  <Database size={12} aria-hidden="true" />
                  Dataset
                </span>
                {dataset.data ? (
                  <Link
                    to="/datasets/$datasetId"
                    params={{ datasetId }}
                    search={{ page: 1 }}
                    title={dataset.data.title}
                  >
                    {dataset.data.title}
                  </Link>
                ) : (
                  <span>
                    {dataset.isPending
                      ? "Loading dataset…"
                      : "Dataset unavailable"}
                  </span>
                )}
                <span className="workspace-surface">/ {surface}</span>
              </div>
            ) : (
              <span>
                Datasets <span className="text-primary">/</span> Library
              </span>
            )}
            <div className="account">
              <span className="account-email">{user.data?.email}</span>
              <Button
                variant="ghost"
                size="sm"
                disabled={logout.isPending}
                onClick={() => logout.mutate({})}
              >
                <LogOut size={14} aria-hidden="true" />
                <span>Log out</span>
              </Button>
            </div>
          </header>
          {dataset.data && <WorkspaceNavigation dataset={dataset.data} />}
        </div>
        <main id="main" className="main-content">
          {logout.error ? <ErrorMessage error={logout.error} /> : null}
          <Outlet />
        </main>
      </div>
    </div>
  );
}
