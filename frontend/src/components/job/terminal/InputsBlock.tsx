// Inputs block for the data-review surface: the analyst's causal question and
// the time column. The question is seeded from what was given on the input page
// and can be refined here after seeing the data; the time column is picked from
// the dataset's real columns (a deterministic detection the analyst can
// correct). Treatment and outcome are no longer chosen here, the analysis
// derives them from the question. Saving persists to the parked job via
// PATCH /jobs/:id/inputs.

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
  const present = value !== '' && columns.includes(value);
  return (
    <div className="flex gap-4 py-1.5 border-b border-edge-subtle/40">
      <span className={labelCls}>{label}</span>
      <div className="flex-1 min-w-0 flex items-center flex-wrap gap-y-1">
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
        {present && <span className="ml-3 text-2xs font-mono text-mint">in dataset</span>}
        {missing && <span className="ml-3 text-2xs font-mono text-rose">not a column</span>}
      </div>
    </div>
  );
}

export function InputsBlock({
  jobId,
  causalQuestion,
  profile,
}: {
  jobId: string | null;
  causalQuestion: string | null;
  profile: ProfileBlock;
}) {
  const queryClient = useQueryClient();
  const columns = profile.data ? Object.keys(profile.data.feature_types ?? {}) : [];
  const detectedTime = profile.data?.has_time_dimension
    ? profile.data.time_column ?? ''
    : '';

  const [question, setQuestion] = useState(causalQuestion ?? '');
  const [selTime, setSelTime] = useState(detectedTime);

  // Re-seed from the persisted values when they change (e.g. after a save
  // refetches the job and the dataset view).
  useEffect(() => setQuestion(causalQuestion ?? ''), [causalQuestion]);
  useEffect(() => setSelTime(detectedTime), [detectedTime]);

  const mutation = useMutation({
    mutationFn: () =>
      updateDatasetInputs(jobId!, {
        causal_question: question.trim(),
        time_column: selTime || null,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['dataset-view', jobId] });
    },
  });

  // Read-only until the schema (columns) is available to pick the time column.
  if (!profile.data) {
    return (
      <p className="text-xs font-mono text-ink-tertiary">
        waiting for the schema to validate inputs…
      </p>
    );
  }

  const dirty =
    question.trim() !== (causalQuestion ?? '').trim() || selTime !== detectedTime;
  const valid =
    question.trim() !== '' && (selTime === '' || columns.includes(selTime));
  const canSave = dirty && valid && !!jobId && !mutation.isPending;

  return (
    <div className="space-y-3">
      <p className="text-2xs font-mono text-ink-tertiary">
        the question you are asking of this dataset; treatment and outcome are derived later
      </p>
      <div>
        <div className="flex gap-4 py-1.5 border-b border-edge-subtle/40">
          <span className={labelCls}>question</span>
          <div className="flex-1 min-w-0">
            <textarea
              aria-label="causal question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={2}
              placeholder="e.g. Does job training increase income?"
              className={
                'w-full ' + selectCls + ' resize-y'
              }
            />
          </div>
        </div>
        <Field
          label="time column"
          value={selTime}
          columns={columns}
          onChange={setSelTime}
          allowNone
        />
      </div>
      <div className="flex items-center gap-3 min-h-[1.75rem]">
        {valid ? (
          <span className="text-2xs font-mono uppercase tracking-[0.15em] text-mint">
            ready to confirm
          </span>
        ) : (
          <span className="text-2xs font-mono uppercase tracking-[0.15em] text-rose">
            a question is required
          </span>
        )}
        {dirty && (
          <button
            type="button"
            onClick={() => mutation.mutate()}
            disabled={!canSave}
            className="text-2xs font-mono uppercase tracking-[0.15em] bg-mint px-3 py-1.5 text-ink-inverse disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {mutation.isPending ? 'saving…' : 'save inputs'}
          </button>
        )}
        {mutation.isError && (
          <span className="text-2xs font-mono text-rose">
            {(mutation.error as { message?: string })?.message || 'save failed'}
          </span>
        )}
      </div>
    </div>
  );
}
