// The plan-confirmation card shown while an analysis run is parked at
// waiting_for_user. Presentational: JobPage owns the POST /jobs/:id/plan
// mutation and passes handlers down; this component only collects edits
// (one labeled input per card item) and a reject reason.
//
// Confirm sends only changed/filled fields, values as strings keyed by the
// item's `field`. Reject opens a one-line reason input and is disabled until
// the reason is non-empty.

import { useState } from 'react';
import type {
  AnalysisPlanGate,
  MethodPlan,
  PlanCardItem,
  SelectedDesign,
} from '../../../services/api';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';
const inputCls =
  'bg-canvas-inset border border-edge-subtle text-ink font-mono text-xs px-2 py-1 ' +
  'focus:outline-none focus:border-amber';

function ItemField({
  item,
  value,
  onChange,
}: {
  item: PlanCardItem;
  value: string;
  onChange: (value: string) => void;
}) {
  const id = `plan-item-${item.field}`;
  const useSelect = item.options.length > 0;
  return (
    <div className="min-w-0">
      <label htmlFor={id} className={`${labelCls} text-ink-secondary block mb-1`}>
        {item.label}
        {item.required && (
          <span className="text-amber ml-1" aria-label="required">
            *
          </span>
        )}
      </label>
      {useSelect ? (
        <select
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`${inputCls} w-full`}
        >
          {(item.current_value === null ||
            !item.options.includes(item.current_value)) && <option value="">—</option>}
          {item.options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`${inputCls} w-full`}
        />
      )}
      <p className="text-2xs text-ink-tertiary mt-0.5">{item.why}</p>
    </div>
  );
}

export function PlanGate({
  gate,
  design,
  methodPlan,
  onConfirm,
  onReject,
  submitting,
  error,
}: {
  gate: AnalysisPlanGate;
  design: SelectedDesign | null;
  methodPlan: MethodPlan | null;
  onConfirm: (edits: Record<string, string>) => void;
  onReject: (reason: string) => void;
  submitting: boolean;
  error: string | null;
}) {
  const card = gate.confirmation_card;
  const items = card?.items ?? [];

  // Field values, prefilled from current_value; user edits land here.
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(items.map((i) => [i.field, i.current_value ?? ''])),
  );
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');

  if (!card) return null;

  // Only changed/filled fields ride the confirm body.
  const buildEdits = (): Record<string, string> => {
    const edits: Record<string, string> = {};
    for (const item of items) {
      const v = (values[item.field] ?? '').trim();
      const original = (item.current_value ?? '').trim();
      if (v !== '' && v !== original) edits[item.field] = v;
    }
    return edits;
  };

  return (
    <div
      className="bg-canvas-raised border border-amber/40 rounded-md p-3 space-y-3"
      data-testid="plan-gate"
    >
      {/* needs-you header */}
      <div className="flex items-center gap-2">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber animate-pulse-live" />
        <span className={`${labelCls} text-amber`}>needs you · confirm the plan</span>
      </div>

      <p className="text-sm text-ink">{card.headline}</p>
      <p className="text-xs text-ink-secondary whitespace-pre-wrap">{card.plan_summary}</p>

      {/* the selected design: label + rationale */}
      {design && (
        <div className="space-y-1">
          <div className="font-mono text-2xs text-ink-tertiary">
            design <span className="text-ink-secondary">{design.design_label}</span>
            {' · '}lane <span className="text-ink-secondary">{design.lane}</span>
            {' · '}confidence{' '}
            <span className="text-ink-secondary">{design.confidence}</span>
          </div>
          <p className="text-xs text-ink-secondary">{design.rationale}</p>
        </div>
      )}

      {/* compact method-plan line */}
      {methodPlan && (
        <div className="font-mono text-2xs text-ink-tertiary">
          plan{' '}
          <span className="text-ink-secondary">
            {[methodPlan.lane, methodPlan.estimator, methodPlan.estimand, methodPlan.outcome].join(
              ' · ',
            )}
          </span>
        </div>
      )}

      {/* why this is parked */}
      {gate.reasons.length > 0 && (
        <ul className="space-y-0.5">
          {gate.reasons.map((r, i) => (
            <li key={i} className="text-2xs font-mono text-amber">
              ! {r}
            </li>
          ))}
        </ul>
      )}

      {/* one labeled input per card item */}
      {items.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {items.map((item) => (
            <ItemField
              key={item.field}
              item={item}
              value={values[item.field] ?? ''}
              onChange={(v) => setValues((prev) => ({ ...prev, [item.field]: v }))}
            />
          ))}
        </div>
      )}

      {error && <p className="text-xs font-mono text-rose">{error}</p>}

      {/* actions */}
      {!rejecting ? (
        <div className="flex items-center gap-3 pt-1">
          <button
            type="button"
            onClick={() => onConfirm(buildEdits())}
            disabled={submitting}
            className={`${labelCls} bg-mint px-4 py-1.5 text-ink-inverse hover:bg-mint-bright disabled:opacity-50`}
          >
            {submitting ? 'submitting…' : 'confirm plan'}
          </button>
          <button
            type="button"
            onClick={() => setRejecting(true)}
            disabled={submitting}
            className={`${labelCls} border border-rose/50 px-3 py-1.5 text-rose hover:bg-rose/10 disabled:opacity-50`}
          >
            reject
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-3 pt-1">
          <input
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="reason for rejecting (required)"
            autoFocus
            className={`${inputCls} w-72 max-w-[45vw]`}
          />
          <button
            type="button"
            onClick={() => {
              setRejecting(false);
              setReason('');
            }}
            className={`${labelCls} px-3 py-1.5 text-ink-tertiary`}
          >
            cancel
          </button>
          <button
            type="button"
            onClick={() => onReject(reason.trim())}
            disabled={submitting || !reason.trim()}
            className={`${labelCls} bg-rose px-4 py-1.5 text-canvas hover:bg-rose/90 disabled:opacity-50`}
          >
            confirm reject
          </button>
        </div>
      )}
    </div>
  );
}
