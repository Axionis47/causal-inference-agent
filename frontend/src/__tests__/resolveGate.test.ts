import { describe, it, expect } from 'vitest';
import { resolveGate } from '../components/job/terminal/resolveGate';
import type { GateSnapshot } from '../services/api';

describe('resolveGate', () => {
  it('returns null when there is no snapshot', () => {
    expect(resolveGate(null)).toBeNull();
    expect(resolveGate(undefined)).toBeNull();
  });

  it('maps a data snapshot to a payload-less data gate', () => {
    const snap: GateSnapshot = { kind: 'data', payload: { files: [] } };
    expect(resolveGate(snap)).toEqual({ kind: 'data' });
  });

  it('maps a dag snapshot to a dag gate carrying its payload', () => {
    const snap: GateSnapshot = {
      kind: 'dag',
      payload: { dag: { adjustment_set: ['x1'] }, justification: {} },
    };
    const gate = resolveGate(snap);
    expect(gate?.kind).toBe('dag');
    expect(gate).toMatchObject({ payload: { dag: { adjustment_set: ['x1'] } } });
  });

  it('maps a results snapshot to a results gate carrying its payload', () => {
    const snap: GateSnapshot = {
      kind: 'results',
      payload: { effects: [{ method: 'ipw', estimate: 1.04 }] },
    };
    const gate = resolveGate(snap);
    expect(gate?.kind).toBe('results');
    expect(gate).toMatchObject({ payload: { effects: [{ method: 'ipw' }] } });
  });
});
