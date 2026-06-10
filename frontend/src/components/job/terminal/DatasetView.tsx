// Dedicated dataset overlay for the live job view (the F1 "data" key in
// JobPage, and the default view on arrival). This is the data-review surface:
// raw rows, a deterministic schema (column types, missingness, time tag), the
// facts Kaggle supplied, and what files were downloaded. Everything here is a
// fact; inferred roles and domain stay user-given / post-approval. useDatasetView
// polls /jobs/:id/dataset until the blocks settle.

import { useEffect, type ReactNode } from 'react';
import type {
  DatasetView as DatasetViewData,
  FileEntry,
  RelationalProfilePayload,
} from '../../../services/api';
import { Caption, StatusLine } from './atoms';
import { InputsBlock } from './InputsBlock';
import { RelationalBlock } from './RelationalBlock';
import { SampleRowsView } from './SampleRowsView';
import { SchemaBlock } from './SchemaBlock';

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

const PLACEHOLDER = <span className="text-ink-tertiary">—</span>;

// One label/value row in the metadata panel. The label column is fixed so
// every field lines up and an empty value still reads as a present-but-blank
// field, not a missing one.
function MetaRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-4 py-1.5 border-b border-edge-subtle/40">
      <span className="w-28 shrink-0 pt-0.5 text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
        {label}
      </span>
      <div className="flex-1 min-w-0 text-xs text-ink-secondary">{children}</div>
    </div>
  );
}

function Chips({ items }: { items: string[] }) {
  if (!items.length) return PLACEHOLDER;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((t) => (
        <span
          key={t}
          className="px-1.5 py-0.5 bg-canvas-raised border border-edge-subtle text-2xs font-mono text-ink-secondary"
        >
          {t}
        </span>
      ))}
    </div>
  );
}

// Render the Kaggle description without a markdown dependency: heading lines
// (#, ##, ###) become small mono captions, everything else is plain wrapped
// text. Keeps the raw "### Context" markers from leaking into the UI.
function Description({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <div className="space-y-1.5 leading-relaxed">
      {lines.map((line, i) => {
        const heading = line.match(/^#{1,6}\s+(.*)$/);
        if (heading) {
          return (
            <p key={i} className="text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary pt-1">
              {heading[1]}
            </p>
          );
        }
        if (!line.trim()) return null;
        return <p key={i} className="text-ink-secondary">{line}</p>;
      })}
    </div>
  );
}

function MetadataBlockView({ view }: { view: DatasetViewData }) {
  const block = view.kaggle_meta;
  if (block.status === 'pending') return <StatusLine label="metadata" status="pending" />;
  if (block.status === 'error')
    return <p className="text-xs text-rose">{block.error || 'metadata fetch failed'}</p>;
  const d = block.data;
  if (!d) return <StatusLine label="metadata" status={block.status} />;

  const cols = Object.entries(d.column_descriptions || {});
  const usability = d.usability_rating != null ? `${Math.round(d.usability_rating * 100)}%` : null;
  const num = (n: number | null | undefined) => (n != null ? n.toLocaleString() : null);

  return (
    <div>
      <MetaRow label="title">{d.title || PLACEHOLDER}</MetaRow>
      <MetaRow label="subtitle">{d.subtitle || PLACEHOLDER}</MetaRow>
      <MetaRow label="description">{d.description ? <Description text={d.description} /> : PLACEHOLDER}</MetaRow>
      <MetaRow label="tags"><Chips items={d.tags} /></MetaRow>
      <MetaRow label="keywords"><Chips items={d.keywords || []} /></MetaRow>
      <MetaRow label="license">{d.license || PLACEHOLDER}</MetaRow>
      <MetaRow label="source">{d.source ? <span className="break-all">{d.source}</span> : PLACEHOLDER}</MetaRow>
      <MetaRow label="size">{d.total_size != null ? fmtBytes(d.total_size) : PLACEHOLDER}</MetaRow>
      <MetaRow label="downloads">{num(d.download_count) ?? PLACEHOLDER}</MetaRow>
      <MetaRow label="votes">{num(d.vote_count) ?? PLACEHOLDER}</MetaRow>
      <MetaRow label="usability">{usability ?? PLACEHOLDER}</MetaRow>
      <MetaRow label="columns">
        {cols.length ? (
          <table className="w-full">
            <tbody className="font-mono">
              {cols.map(([name, desc]) => (
                <tr key={name} className="border-b border-edge-subtle/30">
                  <td className="py-1 pr-3 align-top text-ink">{name}</td>
                  <td className="py-1 text-ink-secondary">{desc || PLACEHOLDER}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <span className="text-ink-tertiary">none provided by source</span>
        )}
      </MetaRow>
    </div>
  );
}

function DownloadBlockView({ view }: { view: DatasetViewData }) {
  const d = view.download;
  return (
    <section>
      <StatusLine label="download" status={d.status} />
      {d.url && <p className="mt-2 font-mono text-2xs text-ink-tertiary break-all">{d.url}</p>}
      {d.error && <p className="mt-2 text-xs text-rose">{d.error}</p>}
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

export function DatasetView({
  view,
  jobId,
  treatment,
  outcome,
  relational,
  onClose,
}: {
  view: DatasetViewData | null;
  jobId: string | null;
  treatment?: string | null;
  outcome?: string | null;
  relational?: RelationalProfilePayload | null;
  onClose: () => void;
}) {
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
              <div className="pt-3"><SampleRowsView view={view} jobId={jobId} /></div>
            </div>
            <div>
              <Caption>[ schema ]</Caption>
              <div className="pt-3"><SchemaBlock block={view.profile} /></div>
            </div>
            <div>
              <Caption>[ inputs ]</Caption>
              <div className="pt-3">
                <InputsBlock
                  jobId={jobId}
                  treatment={treatment ?? null}
                  outcome={outcome ?? null}
                  profile={view.profile}
                />
              </div>
            </div>
            <div>
              <Caption>[ metadata ]</Caption>
              <div className="pt-3"><MetadataBlockView view={view} /></div>
            </div>
            <div>
              <Caption>[ download ]</Caption>
              <div className="pt-3"><DownloadBlockView view={view} /></div>
            </div>
            {relational && relational.files.length > 1 && (
              <div>
                <Caption>[ bundle structure ]</Caption>
                <div className="pt-3"><RelationalBlock profile={relational} /></div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
