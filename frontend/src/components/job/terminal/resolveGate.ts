// Maps a REST gate snapshot (GET /jobs/:id/approval) to the gate-panel union
// JobPage renders. Pure, so it is unit-testable apart from the page. The data
// gate carries no panel payload (its review surface is the dataset view); the
// DAG and results gates carry the same payload shape their SSE events did.

import type { GateSnapshot, DagGatePayload, ResultsGatePayload } from '../../../services/api';

export type ActiveGate =
  | { kind: 'data' }
  | { kind: 'dag'; payload: DagGatePayload }
  | { kind: 'results'; payload: ResultsGatePayload };

export function resolveGate(snapshot: GateSnapshot | null | undefined): ActiveGate | null {
  if (!snapshot) return null;
  if (snapshot.kind === 'dag')
    return { kind: 'dag', payload: snapshot.payload as unknown as DagGatePayload };
  if (snapshot.kind === 'results')
    return { kind: 'results', payload: snapshot.payload as unknown as ResultsGatePayload };
  return { kind: 'data' };
}
