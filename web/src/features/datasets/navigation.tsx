import { useState } from "react";
import { Link } from "@tanstack/react-router";
import {
  ChevronLeft,
  ChevronRight,
  FileSpreadsheet,
  Search,
} from "lucide-react";
import { api } from "@/lib/api/client";
import type { Dataset } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ErrorMessage, Loading } from "@/components/feedback";

export function DatasetNavigation({
  selectedId,
  current,
  onNavigate,
}: {
  selectedId?: string;
  current?: Dataset;
  onNavigate: () => void;
}) {
  const [filter, setFilter] = useState({ page: 1, search: "" });
  const query = api.useQuery("get", "/api/datasets", {
    params: { query: { ...filter, page_size: 20 } },
  });
  const data = query.data;
  const pages = Math.max(1, Math.ceil((data?.total ?? 0) / 20));
  const items = data
    ? [
        ...data.items.filter((dataset) => dataset.id === selectedId),
        ...data.items.filter((dataset) => dataset.id !== selectedId),
      ]
    : [];

  function item(dataset: Dataset) {
    return (
      <Link
        key={dataset.id}
        to="/datasets/$datasetId"
        params={{ datasetId: dataset.id }}
        search={{ page: 1 }}
        className="sidebar-dataset"
        aria-current={selectedId === dataset.id ? "true" : undefined}
        onClick={onNavigate}
      >
        <FileSpreadsheet size={14} aria-hidden="true" />
        <span>
          <strong>{dataset.title}</strong>
          <small title={dataset.version.filename}>
            {dataset.version.filename}
          </small>
        </span>
      </Link>
    );
  }

  return (
    <>
      <div className="sidebar-datasets-heading">
        <h2 className="eyebrow">Datasets</h2>
        <Link
          to="/datasets"
          search={{ page: 1, search: "" }}
          className="context-link"
          aria-current={!selectedId ? "page" : undefined}
          onClick={onNavigate}
        >
          Dataset library
        </Link>
      </div>
      <form
        key={filter.search}
        className="sidebar-search"
        role="search"
        aria-label="Find a dataset"
        onSubmit={(event) => {
          event.preventDefault();
          const value = new FormData(event.currentTarget).get("dataset-search");
          setFilter({
            page: 1,
            search: typeof value === "string" ? value.trim() : "",
          });
        }}
      >
        <Input
          name="dataset-search"
          aria-label="Find a dataset"
          placeholder="Find a dataset…"
          defaultValue={filter.search}
          maxLength={200}
          autoComplete="off"
          type="search"
        />
        <Button
          type="submit"
          variant="ghost"
          size="icon-sm"
          aria-label="Find dataset"
        >
          <Search size={14} aria-hidden="true" />
        </Button>
      </form>
      {filter.search && (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setFilter({ page: 1, search: "" })}
        >
          Clear dataset search
        </Button>
      )}
      {current &&
        data &&
        !data.items.some((dataset) => dataset.id === current.id) && (
          <div className="sidebar-current">
            <p className="eyebrow">Current dataset</p>
            {item(current)}
          </div>
        )}
      <div className="sidebar-dataset-list" aria-busy={query.isFetching}>
        {query.isPending ? (
          <Loading>Loading datasets…</Loading>
        ) : query.error ? (
          <div className="sidebar-feedback">
            <ErrorMessage error={query.error} />
            <Button
              variant="outline"
              size="sm"
              onClick={() => void query.refetch()}
            >
              Retry datasets
            </Button>
          </div>
        ) : data?.items.length ? (
          <nav aria-label="Saved datasets">{items.map(item)}</nav>
        ) : (
          <p className="sidebar-feedback subtext" role="status">
            {filter.search
              ? "No matching datasets. Try another title or clear the search."
              : "No saved datasets yet. Upload a CSV in the dataset library."}
          </p>
        )}
      </div>
      {data && pages > 1 && (
        <div className="sidebar-pagination" aria-label="Dataset list pages">
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Previous dataset page"
            disabled={filter.page <= 1}
            onClick={() => setFilter((old) => ({ ...old, page: old.page - 1 }))}
          >
            <ChevronLeft size={14} aria-hidden="true" />
          </Button>
          <span>
            {data.page} / {pages}
          </span>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Next dataset page"
            disabled={filter.page >= pages}
            onClick={() => setFilter((old) => ({ ...old, page: old.page + 1 }))}
          >
            <ChevronRight size={14} aria-hidden="true" />
          </Button>
        </div>
      )}
    </>
  );
}
