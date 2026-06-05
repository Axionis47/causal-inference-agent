// Dedicated dataset overlay for the live job view (the F1 "data" key in
// JobPage, and the default view on arrival). Shows the raw rows, download
// status, downloaded files, data shape, inferred schema, and Kaggle metadata.
// useDatasetView polls /jobs/:id/dataset until the blocks settle.

import { useEffect } from 'react';
import type {
  DatasetView as DatasetViewData,
  DataProfileSummary,
  FileEntry,
} from '../../../services/api';
import { Caption, StatusLine } from './atoms';
import { SampleRowsView } from './SampleRowsView';

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function roleOf(col: string, p: DataProfileSummary): string {
  if (p.treatment_candidates?.includes(col)) return 'treatment';
  if (p.outcome_candidates?.includes(col)) return 'outcome';
  if (p.potential_confounders?.includes(col)) return 'confounder';
  if (p.potential_instruments?.includes(col)) return 'instrument';
  return '—';
}

const ROLE_TONE: Record<string, string> = {
  treatment: 'text-amber',
  outcome: 'text-mint',
  confounder: 'text-indigo',
  instrument: 'text-bone',
  '—': 'text-ink-tertiary',
};

function DownloadBlockView({ view }: { view: DatasetViewData }) {
  const d = view.download;
  const shape = view.profile.data;
  return (
    <section>
      <StatusLine label="download" status={d.status} />
      {d.url && <p className="mt-2 font-mono text-2xs text-ink-tertiary break-all">{d.url}</p>}
      {d.error && <p className="mt-2 text-xs text-rose">{d.error}</p>}
      {shape && (
        <p className="mt-2 font-mono text-sm text-ink tabular">
          {shape.n_samples.toLocaleString()} rows × {shape.n_features} cols
        </p>
      )}
      {d.files.length > 0 && (
        <table className="mt-3 w-full text-xs">
          <thead>
            <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
              <th className="text-left py-1.5 pr-3">File</th>
              <th className="text-left py-1.5 pr-3">Format</th>
              <th className="text-right py-1.5 pr-3">Size</th>
              <th className="text-left py-1.5">Used</th>
            </tr>
          </thead>
          <tbody className="font-mono text-ink-secondary">
            {d.files.map((f: FileEntry) => (
              <tr key={f.name} className="border-b border-edge-subtle/50">
                <td className={`py-1.5 pr-3 ${f.used ? 'text-ink' : ''}`}>{f.name}</td>
                <td className="py-1.5 pr-3">{f.format}</td>
                <td className="py-1.5 pr-3 text-right tabular">{fmtBytes(f.size_bytes)}</td>
                <td className="py-1.5">
                  {f.used ? <span className="text-mint">yes</span> : <span className="text-ink-tertiary">no</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function Candidates({ p }: { p: DataProfileSummary }) {
  const groups: [string, string[] | undefined][] = [
    ['treatment', p.treatment_candidates],
    ['outcome', p.outcome_candidates],
    ['confounders', p.potential_confounders],
    ['instruments', p.potential_instruments],
  ];
  return (
    <dl className="mt-3 space-y-1.5">
      {groups
        .filter(([, cols]) => cols && cols.length > 0)
        .map(([role, cols]) => (
          <div key={role} className="flex gap-3 text-xs">
            <dt className="text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary w-24 shrink-0 pt-0.5">
              {role}
            </dt>
            <dd className="flex flex-wrap gap-1.5">
              {cols!.map((c) => (
                <span key={c} className="font-mono text-2xs px-2 py-0.5 border border-edge-subtle text-ink-secondary rounded">
                  {c}
                </span>
              ))}
            </dd>
          </div>
        ))}
    </dl>
  );
}

function SchemaBlockView({ view }: { view: DatasetViewData }) {
  const p = view.profile.data;
  return (
    <section>
      <StatusLine label="schema" status={view.profile.status} />
      {view.profile.error && <p className="mt-2 text-xs text-rose">{view.profile.error}</p>}
      {p && <Candidates p={p} />}
      {p && p.feature_types && Object.keys(p.feature_types).length > 0 ? (
        <table className="mt-3 w-full text-xs">
          <thead>
            <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
              <th className="text-left py-1.5 pr-3">Column</th>
              <th className="text-left py-1.5 pr-3">Type</th>
              <th className="text-right py-1.5 pr-3">Missing</th>
              <th className="text-left py-1.5">Role</th>
            </tr>
          </thead>
          <tbody className="font-mono text-ink-secondary">
            {Object.entries(p.feature_types).map(([col, type]) => {
              const miss = p.missing_values?.[col] ?? 0;
              const pct = p.n_samples ? (miss / p.n_samples) * 100 : 0;
              const role = roleOf(col, p);
              return (
                <tr key={col} className="border-b border-edge-subtle/50">
                  <td className="py-1.5 pr-3 text-ink">{col}</td>
                  <td className="py-1.5 pr-3">{type}</td>
                  <td className={`py-1.5 pr-3 text-right tabular ${pct > 20 ? 'text-amber' : ''}`}>
                    {pct.toFixed(0)}%
                  </td>
                  <td className={`py-1.5 ${ROLE_TONE[role]}`}>{role}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        view.profile.status !== 'error' && (
          <p className="mt-2 text-xs text-ink-tertiary">Per-column schema arrives after profiling.</p>
        )
      )}
    </section>
  );
}

function KaggleBlockView({ view }: { view: DatasetViewData }) {
  const k = view.kaggle_meta;
  return (
    <section>
      <StatusLine label="kaggle metadata" status={k.status} />
      {k.error && <p className="mt-2 text-xs text-ink-tertiary">{k.error}</p>}
      {k.data && (
        <div className="mt-3 space-y-3">
          <div className="flex items-center gap-2 flex-wrap">
            {k.data.domain && (
              <span className="text-2xs font-mono uppercase tracking-[0.12em] px-2 py-0.5 border border-edge-subtle text-indigo rounded">
                {k.data.domain}
              </span>
            )}
            {k.data.tags.map((t) => (
              <span key={t} className="text-2xs font-mono px-2 py-0.5 border border-edge-subtle text-ink-tertiary rounded">
                {t}
              </span>
            ))}
          </div>
          {k.data.description && (
            <p className="text-xs text-ink-secondary whitespace-pre-line leading-relaxed">{k.data.description}</p>
          )}
          {Object.keys(k.data.column_descriptions).length > 0 && (
            <dl className="space-y-1">
              {Object.entries(k.data.column_descriptions).map(([col, desc]) => (
                <div key={col} className="flex gap-3 text-xs">
                  <dt className="font-mono text-ink shrink-0">{col}</dt>
                  <dd className="text-ink-tertiary">{desc}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      )}
    </section>
  );
}

export function DatasetView({ view, onClose }: { view: DatasetViewData | null; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  return (
    <div className="terminal fixed inset-0 z-40 bg-canvas flex flex-col">
      <div className="flex items-center justify-between h-10 px-3 bg-canvas-raised border-b border-edge-subtle shrink-0">
        <span className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-tertiary">[ dataset ]</span>
        <button
          onClick={onClose}
          className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-secondary hover:text-ink"
        >
          esc · close
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {!view ? (
          <p className="p-4 text-xs text-ink-tertiary">No dataset information yet.</p>
        ) : (
          <div className="max-w-5xl p-4 space-y-8">
            <div>
              <Caption>[ raw data ]</Caption>
              <div className="pt-3"><SampleRowsView view={view} /></div>
            </div>
            <div>
              <Caption>[ download ]</Caption>
              <div className="pt-3"><DownloadBlockView view={view} /></div>
            </div>
            <div>
              <Caption>[ schema ]</Caption>
              <div className="pt-3"><SchemaBlockView view={view} /></div>
            </div>
            <div>
              <Caption>[ source ]</Caption>
              <div className="pt-3"><KaggleBlockView view={view} /></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
