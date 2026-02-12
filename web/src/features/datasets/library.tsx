import { Link, useNavigate } from "@tanstack/react-router";
import {
  Database,
  FileSpreadsheet,
  ArrowUpRight,
  Search,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { api, fileSize } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ErrorMessage, Loading } from "@/components/feedback";
import { UploadDialog } from "./upload-dialog";

export function DatasetLibrary({
  page,
  search,
}: {
  page: number;
  search: string;
}) {
  const navigate = useNavigate();
  const query = api.useQuery("get", "/api/datasets", {
    params: { query: { page, search, page_size: 20 } },
  });
  const data = query.data;
  function go(nextPage: number) {
    void navigate({ to: "/datasets", search: { search, page: nextPage } });
  }
  return (
    <>
      <div className="eyebrow">01 / DATA INTAKE</div>
      <div className="page-heading">
        <div>
          <h1>
            Dataset library<span className="text-primary">.</span>
          </h1>
          <p className="subtext">
            Your source data, saved and ready to inspect.
          </p>
        </div>
        <UploadDialog />
      </div>
      <div className="metrics">
        <div className="metric">
          <span className="eyebrow">
            {search ? "Matching datasets" : "Saved datasets"}
          </span>
          <span className="metric-value mono-number">
            {data ? String(data.total).padStart(2, "0") : "..."}
          </span>
        </div>
        <div className="metric">
          <span className="eyebrow">Accepted format</span>
          <span className="metric-value">.csv</span>
        </div>
        <div className="metric">
          <span className="eyebrow">Source policy</span>
          <span className="metric-value text-primary">Preserved</span>
        </div>
      </div>
      <section className="panel" aria-labelledby="library-heading">
        <div className="panel-heading">
          <h2 id="library-heading" className="panel-title">
            <Database size={14} aria-hidden="true" />
            {search ? "Search results" : "All datasets"}
          </h2>
          <form
            key={search}
            className="flex gap-2 items-center"
            onSubmit={(event) => {
              event.preventDefault();
              const term = new FormData(event.currentTarget).get("search");
              void navigate({
                to: "/datasets",
                search: {
                  page: 1,
                  search: typeof term === "string" ? term.trim() : "",
                },
              });
            }}
          >
            <Input
              name="search"
              aria-label="Search datasets"
              placeholder="Search by title..."
              defaultValue={search}
              maxLength={200}
              className="h-8 w-44 text-xs"
            />
            <Button
              variant="outline"
              size="icon-sm"
              type="submit"
              aria-label="Search"
            >
              <Search size={14} />
            </Button>
          </form>
        </div>
        {query.isPending ? (
          <Loading>Loading datasets...</Loading>
        ) : query.error ? (
          <div className="p-5">
            <ErrorMessage error={query.error} />
            <Button
              variant="outline"
              className="mt-4"
              onClick={() => void query.refetch()}
            >
              Retry
            </Button>
          </div>
        ) : data && data.items.length > 0 ? (
          <div className="table-scroll">
            <table className="dataset-table">
              <caption className="sr-only">Saved datasets</caption>
              <thead>
                <tr>
                  <th scope="col">DATASET</th>
                  <th scope="col">ROWS</th>
                  <th scope="col">COLUMNS</th>
                  <th scope="col">SIZE</th>
                  <th scope="col">SAVED</th>
                  <th scope="col">
                    <span className="sr-only">Status</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((dataset) => (
                  <tr key={dataset.id}>
                    <td>
                      <Link
                        to="/datasets/$datasetId"
                        params={{ datasetId: dataset.id }}
                        search={{ page: 1 }}
                        className="dataset-name"
                      >
                        <FileSpreadsheet
                          size={19}
                          className="text-primary"
                          aria-hidden="true"
                        />
                        <span>
                          {dataset.title}
                          <span className="filename">
                            {dataset.version.filename}
                          </span>
                        </span>
                        <ArrowUpRight
                          size={14}
                          className="ml-4 text-muted-foreground"
                          aria-hidden="true"
                        />
                      </Link>
                    </td>
                    <td className="mono-number">
                      {dataset.version.row_count.toLocaleString()}
                    </td>
                    <td className="mono-number">
                      {dataset.version.columns.length}
                    </td>
                    <td>{fileSize(dataset.version.size_bytes)}</td>
                    <td>
                      {new Intl.DateTimeFormat("en", {
                        month: "short",
                        day: "2-digit",
                        year: "numeric",
                      }).format(new Date(dataset.created_at))}
                    </td>
                    <td>
                      <span className="tag">
                        v{dataset.version.number} / saved
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">
            <div className="empty-icon">
              <FileSpreadsheet size={25} aria-hidden="true" />
            </div>
            <h2>
              {search
                ? "No matching datasets"
                : "Your first dataset starts here"}
            </h2>
            <p className="subtext">
              {search
                ? "Try a different title or clear your search."
                : "Upload a CSV to explore its rows and columns. Your original file stays intact."}
            </p>
            {search ? (
              <Button
                variant="outline"
                onClick={() =>
                  void navigate({
                    to: "/datasets",
                    search: { page: 1, search: "" },
                  })
                }
              >
                Clear search
              </Button>
            ) : (
              <UploadDialog label="Upload your first CSV" />
            )}
          </div>
        )}
        {data ? (
          <div className="table-footer">
            <span>
              {data.total === 0
                ? "0 datasets"
                : data.items.length === 0
                  ? `No datasets on page ${page}`
                  : `${(page - 1) * data.page_size + 1}-${Math.min(page * data.page_size, data.total)} of ${data.total} datasets`}
            </span>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Previous dataset page"
                disabled={page <= 1}
                onClick={() => go(page - 1)}
              >
                <ChevronLeft size={14} />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Next dataset page"
                disabled={page * data.page_size >= data.total}
                onClick={() => go(page + 1)}
              >
                <ChevronRight size={14} />
              </Button>
            </div>
          </div>
        ) : null}
      </section>
      <div className="workspace-note">
        <ShieldCheck size={15} className="shrink-0 mt-0.5" aria-hidden="true" />
        <span>
          Original files are preserved. Each upload has a stable version and
          checksum.
        </span>
      </div>
    </>
  );
}
