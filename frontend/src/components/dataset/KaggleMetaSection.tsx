import { memo } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { KaggleMetaBlock } from '../../services/api';
import { useExpandable } from './useExpandable';

interface KaggleMetaSectionProps {
  block: KaggleMetaBlock;
}

function truncate(s: string | null | undefined, n: number): string {
  if (!s) return '';
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function KaggleMetaSectionImpl({ block }: KaggleMetaSectionProps) {
  const { expanded, toggle } = useExpandable('kaggle');

  if (block.status === 'error') {
    return (
      <div className="border-t border-ink-100 py-3 animate-fade-in">
        <p className="text-sm text-sig-no">
          Metadata fetch failed{block.error ? ` — ${block.error}` : '.'}
        </p>
      </div>
    );
  }

  const data = block.data;
  if (!data) return null;

  const headlineParts: string[] = [];
  if (data.description) {
    headlineParts.push(truncate(data.description, 80));
  } else {
    headlineParts.push('No description');
  }
  if (data.domain) headlineParts.push(data.domain);

  const omittedFields: string[] = [];
  if (!data.description) omittedFields.push('description');
  if (Object.keys(data.column_descriptions).length === 0) {
    omittedFields.push('per-column descriptions');
  }
  if (data.tags.length === 0) omittedFields.push('tags');

  return (
    <div className="border-t border-ink-100 py-3 animate-fade-in">
      <button
        type="button"
        onClick={toggle}
        className="flex w-full items-center text-left text-sm text-ink-700 hover:text-ink-900"
        aria-expanded={expanded}
      >
        {expanded ? (
          <ChevronDown className="w-4 h-4 mr-2 text-ink-400" />
        ) : (
          <ChevronRight className="w-4 h-4 mr-2 text-ink-400" />
        )}
        <span>Kaggle metadata · {headlineParts.join(' · ')}</span>
      </button>
      {expanded && (
        <div className="mt-3 ml-6 space-y-4">
          {data.description && (
            <p className="text-sm text-ink-700 leading-relaxed whitespace-pre-line">
              {data.description}
            </p>
          )}
          {data.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {data.tags.map((t) => (
                <span
                  key={t}
                  className="badge bg-ink-50 text-ink-600 border border-ink-200"
                >
                  {t}
                </span>
              ))}
            </div>
          )}
          {Object.keys(data.column_descriptions).length > 0 && (
            <div>
              <p className="text-xs uppercase tracking-wider text-ink-500 mb-2">
                Per-column descriptions
              </p>
              <dl className="space-y-2">
                {Object.entries(data.column_descriptions).map(([col, desc]) => (
                  <div key={col} className="text-sm">
                    <dt className="font-mono text-xs text-ink-900 inline">{col}</dt>
                    <dd className="inline text-ink-600"> — {desc}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
          {omittedFields.length > 0 && (
            <p className="text-xs text-ink-400 italic">
              Kaggle did not provide: {omittedFields.join(', ')}.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default memo(KaggleMetaSectionImpl);
