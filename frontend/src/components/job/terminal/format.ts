// Pure formatters for the terminal JobPage. No React, no I/O.
// Tested in isolation: see __tests__/format.test.ts (to be added with first auth slice).

import type { AgentTone } from './agents';

/** HH:MM:SS elapsed since startISO, ticking off nowMs. Returns "--:--:--" if start is missing or invalid. */
export function formatElapsed(startISO: string | undefined, nowMs: number): string {
  if (!startISO) return '--:--:--';
  const startMs = Date.parse(startISO);
  if (Number.isNaN(startMs)) return '--:--:--';
  const sec = Math.max(0, Math.floor((nowMs - startMs) / 1000));
  const h = String(Math.floor(sec / 3600)).padStart(2, '0');
  const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
  const s = String(sec % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

/** Wall-clock HH:MM:SS for an ISO timestamp. Returns "--:--:--" on parse failure. */
export function formatHMS(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '--:--:--';
  return d.toTimeString().slice(0, 8);
}

/** "+5s" / "+1m02s" for the gap between curISO and the chronologically prior event. Empty string when no prior. */
export function formatDelta(curISO: string, priorISO: string | undefined): string {
  if (!priorISO) return '';
  const cur = Date.parse(curISO);
  const prior = Date.parse(priorISO);
  if (Number.isNaN(cur) || Number.isNaN(prior) || cur < prior) return '';
  const sec = Math.floor((cur - prior) / 1000);
  if (sec < 60) return `+${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `+${m}m${s.toString().padStart(2, '0')}`;
}

/** Map a job status string onto the live/settled tone axis. */
export function statusTone(status: string | undefined): AgentTone {
  if (status === 'completed') return 'ok';
  if (status === 'failed') return 'failed';
  if (status === 'cancelled' || status === 'cancelling') return 'pending';
  return 'live';
}

/** 4-char pill label for the top-bar status indicator. */
export function statusPillLabel(status: string | undefined): string {
  if (!status) return '----';
  if (status === 'completed') return 'DONE';
  if (status === 'failed') return 'FAIL';
  if (status === 'cancelled') return 'STOP';
  if (status === 'cancelling') return 'STOP…';
  return 'LIVE';
}
