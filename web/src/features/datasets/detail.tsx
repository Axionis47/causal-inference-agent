import { useMemo } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  type ColumnDef,
} from "@tanstack/react-table";
import {
  ArrowLeft,
  Download,
  ChevronLeft,
  ChevronRight,
  Table2,
  Fingerprint,
} from "lucide-react";
import { api, fileSize } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { ErrorMessage, Loading } from "@/components/feedback";

function PreviewTable({
  columns,
  rows,
  offset,
}: {
  columns: string[];
  rows: string[][];
  offset: number;
}) {
  const definitions = useMemo<ColumnDef<string[]>[]>(
    () => [
      {
        id: "row-index",
        header: "#",
        cell: ({ row }) => offset + row.index + 1,
      },
      ...columns.map((name, index) => ({
        id: String(index),
        header: name,
        accessorFn: (row: string[]) => row[index] ?? "",
        cell: ({ getValue }: { getValue: () => unknown }) => {
          const value = String(getValue());
          return (
            value || (
              <span aria-label="Empty cell" className="text-muted-foreground">
                ∅
              </span>
            )
          );
        },
      })),
    ],
    [columns, offset],
  );
  const table = useReactTable({
    data: rows,
    columns: definitions,
    getCoreRowModel: getCoreRowModel(),
  });
  return (
    <div
      className="table-scroll"
      tabIndex={0}
      role="region"
      aria-label="Scrollable dataset preview"
    >
      <table className="dataset-table preview-table">
        <caption className="sr-only">Original CSV values</caption>
        <thead>
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => (
                <th key={header.id} scope="col">
                  {flexRender(
                    header.column.columnDef.header,
                    header.getContext(),
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id}>
              {row.getVisibleCells().map((cell) => (
                <td
                  key={cell.id}
                  title={
                    cell.column.id === "row-index"
                      ? undefined
                      : String(cell.getValue() ?? "")
                  }
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DatasetDetail({
  datasetId,
  page,
}: {
  datasetId: string;
  page: number;
}) {
  const navigate = useNavigate();
  const dataset = api.useQuery("get", "/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: datasetId } },
  });
  const preview = api.useQuery("get", "/api/datasets/{dataset_id}/preview", {
    params: { path: { dataset_id: datasetId }, query: { page, page_size: 25 } },
  });
  function go(next: number) {
    void navigate({
      to: "/datasets/$datasetId",
      params: { datasetId },
      search: { page: next },
    });
  }
  return (
    <>
      <Link
        to="/datasets"
        search={{ page: 1, search: "" }}
        className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-primary mb-7"
      >
        <ArrowLeft size={13} aria-hidden="true" />
        Dataset library
      </Link>
      {dataset.isPending ? (
        <Loading>Opening dataset...</Loading>
      ) : dataset.error ? (
        <ErrorMessage error={dataset.error} />
      ) : dataset.data ? (
        <>
          <div className="eyebrow">
            SOURCE / VERSION{" "}
            {dataset.data.version.number.toString().padStart(2, "0")}
          </div>
          <div className="page-heading">
            <div>
              <h1 className="break-all">{dataset.data.title}</h1>
              <p className="subtext break-all">
                {dataset.data.version.filename}
              </p>
            </div>
            <Button variant="outline" asChild>
              <a href={`/api/datasets/${datasetId}/file`}>
                <Download size={14} aria-hidden="true" />
                Download original
              </a>
            </Button>
          </div>
          <div className="detail-meta">
            <Link to="/datasets/$datasetId/chat/$versionId" params={{ datasetId, versionId: dataset.data.version.id }} search={{}} className="context-link">Open chat</Link>
            <Link
              to="/datasets/$datasetId/context/$versionId"
              params={{ datasetId, versionId: dataset.data.version.id }}
              search={{ section: "dataset" }}
              className="context-link"
            >
              Open context workspace
            </Link>
            <span>
              ROWS{" "}
              <strong>{dataset.data.version.row_count.toLocaleString()}</strong>
            </span>
            <span>
              COLUMNS <strong>{dataset.data.version.columns.length}</strong>
            </span>
            <span>
              SIZE <strong>{fileSize(dataset.data.version.size_bytes)}</strong>
            </span>
            <span>
              VERSION <strong>v{dataset.data.version.number}</strong>
            </span>
            <span className="text-primary">SOURCE PRESERVED</span>
          </div>
          <section className="panel" aria-labelledby="preview-heading">
            <div className="panel-heading">
              <h2 id="preview-heading" className="panel-title">
                <Table2 size={14} aria-hidden="true" />
                Data preview
              </h2>
              <span className="eyebrow">READ ONLY</span>
            </div>
            {preview.isPending ? (
              <Loading>Reading source rows...</Loading>
            ) : preview.error ? (
              <div className="p-5">
                <ErrorMessage error={preview.error} />
                <Button
                  variant="outline"
                  className="mt-4"
                  onClick={() => void preview.refetch()}
                >
                  Retry preview
                </Button>
              </div>
            ) : preview.data ? (
              <>
                <PreviewTable
                  columns={preview.data.columns}
                  rows={preview.data.rows}
                  offset={(page - 1) * preview.data.page_size}
                />
                <div className="table-footer">
                  <span>
                    {preview.data.rows.length
                      ? `Rows ${(page - 1) * preview.data.page_size + 1}-${Math.min(page * preview.data.page_size, preview.data.total_rows)} of ${preview.data.total_rows.toLocaleString()}`
                      : "No rows on this page"}
                  </span>
                  <div className="flex gap-2 items-center">
                    <span>Page {page}</span>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Previous preview page"
                      disabled={page <= 1}
                      onClick={() => go(page - 1)}
                    >
                      <ChevronLeft size={14} />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Next preview page"
                      disabled={
                        page * preview.data.page_size >= preview.data.total_rows
                      }
                      onClick={() => go(page + 1)}
                    >
                      <ChevronRight size={14} />
                    </Button>
                  </div>
                </div>
              </>
            ) : null}
          </section>
          <details className="mt-6 text-[11px] text-muted-foreground">
            <summary className="cursor-pointer inline-flex items-center gap-2">
              <Fingerprint size={14} aria-hidden="true" />
              Source identity
            </summary>
            <dl className="mt-4 space-y-2 break-all">
              <div>
                <dt className="inline text-foreground">Dataset ID: </dt>
                <dd className="inline">{dataset.data.id}</dd>
              </div>
              <div>
                <dt className="inline text-foreground">Version ID: </dt>
                <dd className="inline">{dataset.data.version.id}</dd>
              </div>
              <div>
                <dt className="inline text-foreground">SHA-256: </dt>
                <dd className="inline">{dataset.data.version.sha256}</dd>
              </div>
            </dl>
          </details>
        </>
      ) : null}
    </>
  );
}
