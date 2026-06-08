import { describe, it, expect } from 'vitest';
import { traceToEvent, tracesToEvents } from '../components/job/terminal/traceEvents';
import type { AgentTrace } from '../services/api';

function trace(partial: Partial<AgentTrace>): AgentTrace {
  return {
    agent_name: 'dag_expert',
    timestamp: '2026-06-08T21:00:00.000Z',
    action: 'step_1_classify_variable_role',
    reasoning: 'because age precedes treatment',
    duration_ms: 1200,
    inputs: {},
    outputs: {},
    tools_called: ['classify_variable_role'],
    token_usage: {},
    ...partial,
  };
}

describe('traceToEvent', () => {
  it('strips the step_N_ prefix from the action', () => {
    const ev = traceToEvent(trace({ action: 'step_19_classify_variable_role' }));
    expect(ev.event_type).toBe('classify_variable_role');
    expect(ev.data.headline).toBe('classify_variable_role');
  });

  it('keeps non-step actions as-is and carries the agent + reasoning', () => {
    const ev = traceToEvent(trace({ action: 'dispatch_to_dag_expert' }));
    expect(ev.event_type).toBe('dispatch_to_dag_expert');
    expect(ev.agent_name).toBe('dag_expert');
    expect(ev.data.reasoning).toBe('because age precedes treatment');
  });
});

describe('tracesToEvents', () => {
  it('returns an empty list for no traces (tape shows awaiting first event)', () => {
    expect(tracesToEvents([])).toEqual([]);
  });

  it('maps every trace and adds one agent_completed marker per agent', () => {
    const traces = [
      trace({ agent_name: 'data_profiler', timestamp: '2026-06-08T21:00:00.000Z', action: 'step_1_x' }),
      trace({ agent_name: 'data_profiler', timestamp: '2026-06-08T21:00:01.000Z', action: 'step_2_y' }),
      trace({ agent_name: 'dag_expert', timestamp: '2026-06-08T21:00:05.000Z', action: 'step_1_z' }),
    ];
    const events = tracesToEvents(traces);
    const completed = events.filter((e) => e.event_type === 'agent_completed');
    expect(completed.map((e) => e.agent_name).sort()).toEqual(['dag_expert', 'data_profiler']);
    // 3 step events + 2 completion markers
    expect(events).toHaveLength(5);
  });

  it('returns events in chronological order with completion after the last step', () => {
    const traces = [
      trace({ agent_name: 'a', timestamp: '2026-06-08T21:00:00.000Z', action: 'step_1_first' }),
      trace({ agent_name: 'a', timestamp: '2026-06-08T21:00:10.000Z', action: 'step_2_last' }),
    ];
    const events = tracesToEvents(traces);
    const times = events.map((e) => e.timestamp);
    const sorted = [...times].sort((x, y) => x.localeCompare(y));
    expect(times).toEqual(sorted);
    // the completion marker is the final event
    expect(events[events.length - 1].event_type).toBe('agent_completed');
  });
});
