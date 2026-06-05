// Raw-data preview block for the dataset overlay. A Kaggle bundle can ship
// several files, so this shows a dropdown of every downloaded tabular file
// (the loaded one marked "used") above a single scrollable table of that
// file's first rows. Fed by the `sample` block of GET /jobs/:id/dataset.

import { useState } from 'react';
import type { DatasetView as DatasetViewData } from '../../../services/api';
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

export function SampleRowsView({ view }: { view: DatasetViewData }) {
  const s = view.sample;
  const files = s.files;
  const [picked, setPicked] = useState<string | null>(null);

  if (s.status === 'pending') {
    return (
      <section>
        <StatusLine label="rows" status={s.status} />
        <p className="mt-2 text-xs text-ink-tertiary">Sample appears once the files are downloaded.</p>
      </section>
    );
  }
  if (files.length === 0) {
    return (
      <section>
        <StatusLine label="rows" status={s.status} />
        <p className="mt-2 text-xs text-ink-tertiary">No previewable tabular files in this bundle.</p>
      </section>
    );
  }

  const current =
    files.find((f) => f.name === picked) ?? files.find((f) => f.used) ?? files[0];

  return (
    <section>
      <StatusLine label="rows" status={s.status} />
      <div className="mt-3 flex items-center gap-3 flex-wrap">
        <label className="text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">file</label>
        <select value={current.name} onChange={(e) => setPicked(e.target.value)} className={selectCls}>
          {files.map((f) => (
            <option key={f.name} value={f.name}>
              {f.name}
              {f.used ? '  (used)' : ''}
            </option>
          ))}
        </select>
        <span className="text-2xs font-mono text-ink-tertiary tabular">
          {current.error
            ? 'read failed'
            : `first ${current.rows.length} of ${(current.total_rows ?? current.rows.length).toLocaleString()} rows`}
        </span>
      </div>

      {current.error ? (
        <p className="mt-2 text-xs text-rose">{current.error}</p>
      ) : (
        <div className="mt-3 max-h-[60vh] overflow-auto border border-edge-subtle">
          <table className="w-full text-xs border-collapse">
            <thead className="sticky top-0 z-10 bg-canvas-raised">
              <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
                {current.columns.map((c) => (
                  <th key={c} className="text-left py-1.5 px-3 whitespace-nowrap">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono text-ink-secondary">
              {current.rows.map((row, i) => (
                <tr key={i} className="border-b border-edge-subtle/50">
                  {current.columns.map((c) => (
                    <td key={c} className="py-1.5 px-3 whitespace-nowrap tabular">{fmtCell(row[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
