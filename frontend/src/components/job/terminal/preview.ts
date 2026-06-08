// Synthetic mid-pipeline state for /jobs/__preview. Lets the layout be reviewed
// without a real backend job. Effect estimator is "live" so the live/settled axis
// shows both tones at once.

import type { AgentEvent, JobDetail } from '../../../services/api';

export function buildPreviewState(): { job: JobDetail; events: AgentEvent[] } {
  const now = Date.now();
  const stamp = (offsetSec: number) =>
    new Date(now - offsetSec * 1000).toISOString();

  const events: AgentEvent[] = [
    { timestamp: stamp(240), agent_name: 'dataset_inspector', event_type: 'agent_started', data: {} },
    { timestamp: stamp(232), agent_name: 'dataset_inspector', event_type: 'agent_completed', data: { headline: 'picked train.csv · 200k rows · 18 cols' } },
    { timestamp: stamp(228), agent_name: 'data_profiler', event_type: 'agent_started', data: {} },
    { timestamp: stamp(210), agent_name: 'data_profiler', event_type: 'agent_completed', data: { headline: 'treatment=treatment, outcome=re78, 3 confounder candidates' } },
    { timestamp: stamp(208), agent_name: 'domain_knowledge', event_type: 'agent_started', data: {} },
    { timestamp: stamp(195), agent_name: 'domain_knowledge', event_type: 'agent_completed', data: { headline: 'labour-economics priors loaded' } },
    { timestamp: stamp(193), agent_name: 'eda_agent', event_type: 'agent_started', data: {} },
    { timestamp: stamp(170), agent_name: 'eda_agent', event_type: 'agent_completed', data: { headline: '2 confounders flagged: age, education' } },
    { timestamp: stamp(168), agent_name: 'causal_discovery', event_type: 'agent_started', data: {} },
    { timestamp: stamp(150), agent_name: 'causal_discovery', event_type: 'agent_completed', data: { headline: 'NOTEARS DAG, 11 edges' } },
    { timestamp: stamp(148), agent_name: 'dag_expert', event_type: 'agent_started', data: {} },
    { timestamp: stamp(130), agent_name: 'dag_expert', event_type: 'agent_completed', data: { headline: 'adjustment set: {age, educ, married}' } },
    { timestamp: stamp(128), agent_name: 'ps_diagnostics', event_type: 'agent_started', data: {} },
    { timestamp: stamp(110), agent_name: 'ps_diagnostics', event_type: 'agent_completed', data: { headline: 'overlap ok · SMD < 0.1 on 8/9 covariates' } },
    { timestamp: stamp(108), agent_name: 'effect_estimator', event_type: 'agent_started', data: {} },
    { timestamp: stamp(72),  agent_name: 'effect_estimator', event_type: 'agent_started', data: { headline: 'tool: get_dag_adjustment_set' } },
    { timestamp: stamp(28),  agent_name: 'effect_estimator', event_type: 'agent_started', data: { headline: 'DML running · 2/3 methods done' } },
  ];

  const job: JobDetail = {
    id: '7a3f1c2e-preview',
    status: 'estimating_effects',
    progress_percentage: 64,
    current_agent: 'effect_estimator',
    kaggle_url: 'kaggle://lalonde-nsw',
    treatment_variable: 'treatment',
    outcome_variable: 're78',
    iteration_count: 1,
    created_at: stamp(248),
    updated_at: stamp(0),
  };

  return { job, events };
}
