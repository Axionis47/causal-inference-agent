// Presentable view of a JSON artifact: a key/value tree instead of raw text.
// Bespoke per-schema views (spec cards, eligibility matrices) layer on top of
// the directory later; this is the fallback every JSON artifact gets.

import type { ReactNode } from 'react';

function isPrimitive(v: unknown): boolean {
  return v === null || ['string', 'number', 'boolean'].includes(typeof v);
}

function Primitive({ value }: { value: unknown }) {
  if (value === null) return <span className="text-ink-tertiary">null</span>;
  if (typeof value === 'string') return <span className="text-ink">{value}</span>;
  if (typeof value === 'boolean')
    return <span className={value ? 'text-mint' : 'text-rose'}>{String(value)}</span>;
  return <span className="text-indigo tabular">{String(value)}</span>;
}

function Node({ name, value }: { name: string; value: unknown }): ReactNode {
  if (isPrimitive(value)) {
    return (
      <div className="flex gap-2 min-w-0">
        <span className="text-ink-tertiary shrink-0">{name}</span>
        <span className="min-w-0 break-words">
          <Primitive value={value} />
        </span>
      </div>
    );
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return (
        <div className="flex gap-2">
          <span className="text-ink-tertiary">{name}</span>
          <span className="text-ink-tertiary">[]</span>
        </div>
      );
    }
    if (value.every(isPrimitive)) {
      return (
        <div className="flex gap-2 min-w-0">
          <span className="text-ink-tertiary shrink-0">{name}</span>
          <span className="min-w-0 break-words text-ink">
            {value.map((v, i) => (
              <span key={i}>
                {i > 0 && <span className="text-ink-tertiary">, </span>}
                <Primitive value={v} />
              </span>
            ))}
          </span>
        </div>
      );
    }
    return (
      <div>
        <span className="text-ink-tertiary">{name}</span>
        <div className="pl-3 border-l border-edge-subtle/40 ml-0.5 space-y-0.5">
          {value.map((v, i) => (
            <Node key={i} name={`[${i}]`} value={v} />
          ))}
        </div>
      </div>
    );
  }
  const entries = Object.entries(value as Record<string, unknown>);
  return (
    <div>
      <span className="text-ink-tertiary">{name}</span>
      <div className="pl-3 border-l border-edge-subtle/40 ml-0.5 space-y-0.5">
        {entries.length === 0 ? (
          <span className="text-ink-tertiary">{'{}'}</span>
        ) : (
          entries.map(([k, v]) => <Node key={k} name={k} value={v} />)
        )}
      </div>
    </div>
  );
}

export function JsonView({ text }: { text: string }) {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    // not actually JSON; show the text rather than nothing
    return (
      <pre className="font-mono text-2xs text-ink-secondary whitespace-pre-wrap">{text}</pre>
    );
  }
  if (isPrimitive(value)) {
    return (
      <div className="font-mono text-2xs">
        <Primitive value={value} />
      </div>
    );
  }
  const entries = Array.isArray(value)
    ? value.map((v, i) => [`[${i}]`, v] as const)
    : Object.entries(value as Record<string, unknown>);
  return (
    <div className="font-mono text-2xs leading-relaxed space-y-0.5">
      {entries.map(([k, v]) => (
        <Node key={k} name={k} value={v} />
      ))}
    </div>
  );
}
