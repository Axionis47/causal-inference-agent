// Bundle structure block for the data-review surface: how the downloaded files
// relate (shape, per-file rows/cols, candidate id columns, shared schemas).
// Read-only and structural; one file still feeds the analysis. Renders nothing
// for a single-file bundle, where there is no structure to report.

import type { RelationalProfilePayload } from '../../../services/api';

const SHAPE_LABEL: Record<RelationalProfilePayload['shape_hint'], string> = {
  single: 'single file',
  same_schema_shards: 'same-schema shards — these files share one schema, like pieces of one table',
  multiple_files: 'multiple files — some share a schema, some differ',
  unrelated: 'unrelated files — no two share a schema',
};

export function RelationalBlock({ profile }: { profile: RelationalProfilePayload | null }) {
  if (!profile || profile.files.length <= 1) return null;

  return (
    <section>
      <p className="text-xs font-mono text-ink-secondary">{SHAPE_LABEL[profile.shape_hint]}</p>
      <p className="mt-1 text-2xs font-mono text-ink-tertiary">{profile.note}</p>

      <table className="mt-3 w-full text-xs">
        <thead>
          <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
            <th className="text-left py-1.5 pr-3">File</th>
            <th className="text-right py-1.5 pr-3">Rows</th>
            <th className="text-right py-1.5 pr-3">Cols</th>
            <th className="text-left py-1.5">Candidate id columns</th>
          </tr>
        </thead>
        <tbody className="font-mono text-ink-secondary">
          {profile.files.map((f) => (
            <tr key={f.file} className="border-b border-edge-subtle/50">
              <td className="py-1.5 pr-3 text-ink">{f.file}</td>
              <td className="py-1.5 pr-3 text-right tabular">{f.n_rows.toLocaleString()}</td>
              <td className="py-1.5 pr-3 text-right tabular">{f.n_cols}</td>
              <td className="py-1.5">
                {f.key_candidates.length ? (
                  f.key_candidates.join(', ')
                ) : (
                  <span className="text-ink-tertiary">none</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {profile.same_schema_groups.length > 0 && (
        <div className="mt-3 space-y-1">
          {profile.same_schema_groups.map((group, i) => (
            <p key={i} className="text-2xs font-mono text-ink-tertiary">
              shared schema: {group.join(', ')}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
