// Raw-data preview for the dataset overlay. A Kaggle bundle can ship several
// files, so this shows a dropdown of every downloaded file (the loaded one
// marked "used") above a scrollable table. The file list and column names
// come from the dataset view's manifest-backed download.files; the rows are
// fetched a page at a time from GET /jobs/:id/dataset/files/:name/rows, so a
// dataset of any size is paged through without loading it all.

import { useEffect, useMemo, useState } from 'react';
import type { DatasetView as DatasetViewData } from '../../../services/api';
import { DATASET_ROWS_PAGE_SIZE } from '../../../config/constants';
import { useDatasetRows } from '../../../hooks/useDatasetRows';
import { StatusLine } from './atoms';

function fmtCell(v: unknown): string {
  if (v === null || v === undefined || v === '') return '·';
  if (typeof v === 'number') {
    return Number.isInteger(v)
      ? String(v)
      : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  }
  return String(v);
}

const selectCls =
  'bg-canvas-inset border border-edge-subtle text-ink font-mono text-xs px-2 py-1 ' +
  'focus:outline-none focus:border-amber max-w-[60%] truncate';

const pageBtnCls =
  'text-2xs font-mono uppercase tracking-[0.12em] px-2 py-1 border border-edge-subtle ' +
  'text-ink-secondary hover:border-amber disabled:opacity-40 disabled:hover:border-edge-subtle';

export function SampleRowsView({
  view,
  jobId,
}: {
  view: DatasetViewData;
  jobId: string | null;
}) {
  const files = view.download.files;
  const limit = DATASET_ROWS_PAGE_SIZE;
  const [picked, setPicked] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);

  const current = useMemo(
    () =>
      files.find((f) => f.name === picked) ??
      files.find((f) => f.used && f.tabular !== false) ??
      files.find((f) => f.tabular !== false) ??
      files[0],
    [files, picked]
  );
  // Non-tabular files (images, pdfs, freeform text) are listed but have no
  // normalised parquet, so there are no rows to fetch or show.
  const previewable = current?.tabular !== false;

  // A new file starts at its first page.
  useEffect(() => {
    setOffset(0);
  }, [current?.name]);

  const { page, isLoading, error } = useDatasetRows(
    jobId,
    previewable ? current?.name ?? null : null,
    offset,
    limit
  );

  if (files.length === 0) {
    const downloading =
      view.download.status === 'pending' || view.download.status === 'downloading';
    return (
      <section>
        <StatusLine label="rows" status={view.download.status} />
        <p className="mt-2 text-xs text-ink-tertiary">
          {downloading
            ? 'Rows appear once the files are downloaded.'
            : 'No previewable files in this bundle.'}
        </p>
      </section>
    );
  }

  // Column names are known from the manifest before any rows load; the page's
  // own columns take over once it arrives.
  const columns = page?.columns?.length ? page.columns : current?.columns ?? [];
  const total = current?.n_rows ?? page?.total_rows ?? null;
  const rows = page?.rows ?? [];
  const from = total === 0 ? 0 : offset + 1;
  const to = total != null ? Math.min(offset + limit, total) : offset + rows.length;
  const canPrev = offset > 0;
  const canNext = total != null ? offset + limit < total : rows.length === limit;

  return (
    <section>
      <StatusLine label="rows" status={view.download.status} />
      <div className="mt-3 flex items-center gap-3 flex-wrap">
        <label className="text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">file</label>
        <select
          value={current?.name}
          onChange={(e) => setPicked(e.target.value)}
          className={selectCls}
        >
          {files.map((f) => (
            <option key={f.name} value={f.name}>
              {f.name}
              {f.tabular === false ? '  (no preview)' : f.used ? '  (used)' : ''}
            </option>
          ))}
        </select>
        <span className="text-2xs font-mono text-ink-tertiary tabular">
          {!previewable
            ? `${current?.format ?? 'binary'} · not previewable`
            : error
              ? 'read failed'
              : total != null
                ? `rows ${from.toLocaleString()}-${to.toLocaleString()} of ${total.toLocaleString()}`
                : `rows ${from}-${to}`}
        </span>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            className={pageBtnCls}
            disabled={!canPrev}
            onClick={() => setOffset((o) => Math.max(o - limit, 0))}
          >
            prev
          </button>
          <button
            type="button"
            className={pageBtnCls}
            disabled={!canNext}
            onClick={() => setOffset((o) => o + limit)}
          >
            next
          </button>
        </div>
      </div>

      {!previewable ? (
        <p className="mt-3 text-xs text-ink-tertiary">
          This file is not tabular, so it has no row preview. It is still part of
          the downloaded bundle and is listed in the download section.
        </p>
      ) : error ? (
        <p className="mt-2 text-xs text-rose">{error}</p>
      ) : (
        <div className="mt-3 max-h-[60vh] overflow-auto border border-edge-subtle">
          <table className="w-full text-xs border-collapse">
            <thead className="sticky top-0 z-10 bg-canvas-raised">
              <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
                {columns.map((c) => (
                  <th key={c} className="text-left py-1.5 px-3 whitespace-nowrap">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono text-ink-secondary">
              {rows.map((row, i) => (
                <tr key={i} className="border-b border-edge-subtle/50">
                  {columns.map((c) => (
                    <td key={c} className="py-1.5 px-3 whitespace-nowrap tabular">{fmtCell(row[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {isLoading && rows.length === 0 && (
            <p className="p-3 text-xs text-ink-tertiary">Loading rows...</p>
          )}
        </div>
      )}
    </section>
  );
}
