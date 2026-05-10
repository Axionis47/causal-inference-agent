import { memo, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { ProfileBlock } from '../../services/api';
import { useExpandable } from './useExpandable';

type SortKey = 'name' | 'dtype' | 'missing';
type SortDir = 'asc' | 'desc';

interface ColumnRow {
  name: string;
  dtype: string;
  missingPct: number;
  missingCount: number;
  role: 'treatment' | 'outcome' | 'confounder' | 'instrument' | null;
}

interface SchemaSectionProps {
  block: ProfileBlock;
}

function SchemaSectionImpl({ block }: SchemaSectionProps) {
  const { expanded, toggle } = useExpandable('schema');
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  // Hooks must run unconditionally — derive rows from block.data which
  // may be null. Early returns happen AFTER all hooks have been called.
  const data = block.data;

  const rows: ColumnRow[] = useMemo(() => {
    if (!data) return [];
    // feature_types / missing_values may be absent on the persisted
    // path (storage layer only saves a profile summary). Default to
    // empty so the schema table degrades gracefully — headline still
    // renders, just with no per-column rows.
    const featureTypes = data.feature_types ?? {};
    const missingValues = data.missing_values ?? {};
    const treatmentSet = new Set(data.treatment_candidates ?? []);
    const outcomeSet = new Set(data.outcome_candidates ?? []);
    const confounderSet = new Set(data.potential_confounders ?? []);
    const instrumentSet = new Set(data.potential_instruments ?? []);

    return Object.entries(featureTypes).map(([name, dtype]) => {
      const missingCount = missingValues[name] ?? 0;
      const missingPct = data.n_samples > 0
        ? (missingCount / data.n_samples) * 100
        : 0;
      const role: ColumnRow['role'] = treatmentSet.has(name)
        ? 'treatment'
        : outcomeSet.has(name)
          ? 'outcome'
          : confounderSet.has(name)
            ? 'confounder'
            : instrumentSet.has(name)
              ? 'instrument'
              : null;
      return { name, dtype, missingPct, missingCount, role };
    });
  }, [data]);

  const sortedRows = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'name') cmp = a.name.localeCompare(b.name);
      else if (sortKey === 'dtype') cmp = a.dtype.localeCompare(b.dtype);
      else if (sortKey === 'missing') cmp = a.missingPct - b.missingPct;
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  if (block.status === 'error') {
    return (
      <div className="border-t border-ink-100 py-3 animate-fade-in">
        <p className="text-sm text-sig-no">
          Profile failed{block.error ? ` — ${block.error}` : '.'}
        </p>
      </div>
    );
  }

  if (!data) return null;

  const avgMissing = rows.length
    ? rows.reduce((s, r) => s + r.missingPct, 0) / rows.length
    : 0;

  const tParts: string[] = [];
  if ((data.treatment_candidates ?? []).length > 0) {
    tParts.push(`T: ${data.treatment_candidates[0]}`);
  }
  if ((data.outcome_candidates ?? []).length > 0) {
    tParts.push(`Y: ${data.outcome_candidates[0]}`);
  }

  const headline = [
    `${data.n_samples.toLocaleString()} rows × ${data.n_features} columns`,
    `${avgMissing.toFixed(1)}% avg missing`,
    ...tParts,
  ].join(' · ');

  function onHeaderClick(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  }

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
        <span>Schema · {headline}</span>
      </button>
      {expanded && (
        <div className="mt-3 ml-6">
          <table className="journal-table">
            <thead>
              <tr>
                <SortHeader
                  label="Column"
                  k="name"
                  sortKey={sortKey}
                  sortDir={sortDir}
                  onClick={onHeaderClick}
                />
                <SortHeader
                  label="Type"
                  k="dtype"
                  sortKey={sortKey}
                  sortDir={sortDir}
                  onClick={onHeaderClick}
                />
                <SortHeader
                  label="Missing"
                  k="missing"
                  sortKey={sortKey}
                  sortDir={sortDir}
                  onClick={onHeaderClick}
                />
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((r) => (
                <tr key={r.name}>
                  <td className="data-value">{r.name}</td>
                  <td className="text-xs text-ink-500">{r.dtype}</td>
                  <td
                    data-testid={`missing-${r.name}`}
                    className={
                      r.missingPct > 20
                        ? 'data-value text-amber-700'
                        : 'data-value'
                    }
                  >
                    {r.missingPct.toFixed(1)}%
                  </td>
                  <td>
                    <RoleBadge role={r.role} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

interface SortHeaderProps {
  label: string;
  k: SortKey;
  sortKey: SortKey;
  sortDir: SortDir;
  onClick: (k: SortKey) => void;
}
function SortHeader({ label, k, sortKey, sortDir, onClick }: SortHeaderProps) {
  const isActive = sortKey === k;
  return (
    <th>
      <button
        type="button"
        onClick={() => onClick(k)}
        className="text-left font-semibold text-ink-900 text-xs uppercase tracking-wider hover:text-accent"
      >
        {label}
        {isActive ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
      </button>
    </th>
  );
}

function RoleBadge({ role }: { role: ColumnRow['role'] }) {
  if (!role) return <span className="text-xs text-ink-400">—</span>;
  const styles: Record<string, string> = {
    treatment: 'bg-blue-50 text-accent border border-blue-200',
    outcome: 'bg-blue-50 text-accent border border-blue-200',
    confounder: 'bg-ink-50 text-ink-600 border border-ink-200',
    instrument: 'bg-ink-50 text-ink-600 border border-ink-200',
  };
  return <span className={`badge ${styles[role]} capitalize`}>{role}</span>;
}

export default memo(SchemaSectionImpl);
