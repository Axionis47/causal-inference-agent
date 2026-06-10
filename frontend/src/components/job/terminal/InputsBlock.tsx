// Inputs block for the data-review surface: the analyst's treatment, outcome,
// and time column, picked from the dataset's real columns. Seeded from what was
// given on the input page (treatment/outcome) and the detected time column; a
// mismatch such as a typo is flagged and corrected here. Saving persists to the
// parked job via PATCH /jobs/:id/inputs.

import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { ProfileBlock } from '../../../services/api';
import { updateDatasetInputs } from '../../../services/api';

const labelCls =
  'w-28 shrink-0 pt-1.5 text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary';
const selectCls =
  'bg-canvas-inset border border-edge-subtle text-ink font-mono text-xs px-2 py-1 ' +
  'focus:outline-none focus:border-edge-strong';

function Field({
  label,
  value,
  columns,
  onChange,
  allowNone,
}: {
  label: string;
  value: string;
  columns: string[];
  onChange: (v: string) => void;
  allowNone?: boolean;
}) {
  const missing = value !== '' && !columns.includes(value);
  return (
    <div className="flex gap-4 py-1.5 border-b border-edge-subtle/40">
      <span className={labelCls}>{label}</span>
      <div className="flex-1 min-w-0">
        <select
          aria-label={label}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={selectCls}
        >
          {allowNone && <option value="">none</option>}
          {missing && <option value={value}>{value} (not a column)</option>}
          {columns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        {missing && <span className="ml-3 text-2xs font-mono text-rose">not a column</span>}
      </div>
    </div>
  );
}

export function InputsBlock({
  jobId,
  treatment,
  outcome,
  profile,
}: {
  jobId: string | null;
  treatment: string | null;
  outcome: string | null;
  profile: ProfileBlock;
}) {
  const queryClient = useQueryClient();
  const columns = profile.data ? Object.keys(profile.data.feature_types ?? {}) : [];
  const detectedTime = profile.data?.has_time_dimension
    ? profile.data.time_column ?? ''
    : '';

  const [selT, setSelT] = useState(treatment ?? '');
  const [selY, setSelY] = useState(outcome ?? '');
  const [selTime, setSelTime] = useState(detectedTime);

  // Re-seed from the persisted values when they change (e.g. after a save
  // refetches the job and the dataset view).
  useEffect(() => setSelT(treatment ?? ''), [treatment]);
  useEffect(() => setSelY(outcome ?? ''), [outcome]);
  useEffect(() => setSelTime(detectedTime), [detectedTime]);

  const mutation = useMutation({
    mutationFn: () =>
      updateDatasetInputs(jobId!, {
        treatment_variable: selT,
        outcome_variable: selY,
        time_column: selTime || null,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['dataset-view', jobId] });
    },
  });

  // Read-only until the schema (columns) is available to pick from.
  if (!profile.data) {
    return (
      <p className="text-xs font-mono text-ink-tertiary">
        waiting for the schema to validate inputs…
      </p>
    );
  }

  const dirty =
    selT !== (treatment ?? '') ||
    selY !== (outcome ?? '') ||
    selTime !== detectedTime;
  const valid =
    columns.includes(selT) &&
    columns.includes(selY) &&
    (selTime === '' || columns.includes(selTime));
  const canSave = dirty && valid && !!jobId && !mutation.isPending;

  return (
    <div className="space-y-3">
      <div>
        <Field label="treatment" value={selT} columns={columns} onChange={setSelT} />
        <Field label="outcome" value={selY} columns={columns} onChange={setSelY} />
        <Field
          label="time column"
          value={selTime}
          columns={columns}
          onChange={setSelTime}
          allowNone
        />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => mutation.mutate()}
          disabled={!canSave}
          className="text-2xs font-mono uppercase tracking-[0.15em] bg-mint px-3 py-1.5 text-ink-inverse disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {mutation.isPending ? 'saving…' : 'save inputs'}
        </button>
        {mutation.isError && (
          <span className="text-2xs font-mono text-rose">
            {(mutation.error as { message?: string })?.message || 'save failed'}
          </span>
        )}
      </div>
    </div>
  );
}
