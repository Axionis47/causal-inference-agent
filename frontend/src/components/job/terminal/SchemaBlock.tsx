// Schema block for the data-review surface: the deterministic structural
// profile (column types, missingness, and a time tag) computed before the gate.
// Facts only. Inferred treatment/outcome roles are not shown here; they stay
// user-given. Shows a status line until the profile loads.

import type { ProfileBlock } from '../../../services/api';
import { StatusLine } from './atoms';

export function SchemaBlock({ block }: { block: ProfileBlock }) {
  if (block.status === 'error')
    return <p className="text-xs text-rose">{block.error || 'profile failed'}</p>;
  const d = block.data;
  if (!d) return <StatusLine label="schema" status={block.status} />;

  const featureTypes = d.feature_types ?? {};
  const missing = d.missing_values ?? {};
  const columns = Object.keys(featureTypes);

  return (
    <section>
      <StatusLine label="schema" status={block.status} />
      <p className="mt-2 text-2xs font-mono text-ink-tertiary">
        {d.n_samples.toLocaleString()} rows · {d.n_features} columns
        {d.has_time_dimension && d.time_column ? ` · time: ${d.time_column}` : ''}
      </p>
      {columns.length > 0 && (
        <table className="mt-3 w-full text-xs">
          <thead>
            <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
              <th className="text-left py-1.5 pr-3">Column</th>
              <th className="text-left py-1.5 pr-3">Type</th>
              <th className="text-right py-1.5">Missing</th>
            </tr>
          </thead>
          <tbody className="font-mono text-ink-secondary">
            {columns.map((name) => {
              const pct =
                d.n_samples > 0 ? ((missing[name] ?? 0) / d.n_samples) * 100 : 0;
              const isTime = d.time_column === name;
              return (
                <tr key={name} className="border-b border-edge-subtle/50">
                  <td className={`py-1.5 pr-3 ${isTime ? 'text-mint' : 'text-ink'}`}>
                    {name}
                  </td>
                  <td className="py-1.5 pr-3">{featureTypes[name]}</td>
                  <td
                    data-testid={`missing-${name}`}
                    className={`py-1.5 text-right tabular ${pct > 20 ? 'text-amber' : ''}`}
                  >
                    {pct.toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}
