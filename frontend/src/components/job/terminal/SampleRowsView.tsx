// Raw-data preview block for the dataset overlay: the first N real rows of
// the downloaded file, so the analyst sees actual cell values, not just the
// schema. Fed by the `sample` block of GET /jobs/:id/dataset.

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

export function SampleRowsView({ view }: { view: DatasetViewData }) {
  const s = view.sample;

  if (s.status === 'pending') {
    return (
      <section>
        <StatusLine label="rows" status={s.status} />
        <p className="mt-2 text-xs text-ink-tertiary">Sample appears once the file is loaded.</p>
      </section>
    );
  }

  if (s.status !== 'loaded' || s.rows.length === 0) {
    return (
      <section>
        <StatusLine label="rows" status={s.status} />
        {s.error && <p className="mt-2 text-xs text-rose">{s.error}</p>}
      </section>
    );
  }

  return (
    <section>
      <StatusLine label="rows" status={s.status} />
      <p className="mt-2 font-mono text-2xs text-ink-tertiary uppercase tracking-[0.12em]">
        first {s.rows.length} of {(s.total_rows ?? s.rows.length).toLocaleString()} rows
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
              {s.columns.map((c) => (
                <th key={c} className="text-left py-1.5 pr-4 whitespace-nowrap">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono text-ink-secondary">
            {s.rows.map((row, i) => (
              <tr key={i} className="border-b border-edge-subtle/50">
                {s.columns.map((c) => (
                  <td key={c} className="py-1.5 pr-4 whitespace-nowrap tabular">{fmtCell(row[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
