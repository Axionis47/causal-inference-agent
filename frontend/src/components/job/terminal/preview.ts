// Synthetic mid-pipeline state for /jobs/__preview. Lets the layout be reviewed
// without a real backend job. Effect estimator is "live" so the live/settled axis
// shows both tones at once.

import type { AgentEvent, AgentTrace, JobDetail } from '../../../services/api';

export function buildPreviewState(): { job: JobDetail; events: AgentEvent[]; traces: AgentTrace[] } {
  const now = Date.now();
  const stamp = (offsetSec: number) =>
    new Date(now - offsetSec * 1000).toISOString();

  const events: AgentEvent[] = [
    { timestamp: stamp(240), agent_name: 'dataset_inspector', event_type: 'agent_started', data: {} },
    { timestamp: stamp(232), agent_name: 'dataset_inspector', event_type: 'agent_completed', data: { headline: 'picked train.csv · 200k rows · 18 cols' } },
    {
      timestamp: stamp(231),
      agent_name: 'dataset_inspector',
      event_type: 'dataset_inspection_complete',
      data: {
        selected_file: 'train.csv',
        relational_profile: {
          shape_hint: 'same_schema_shards',
          files: [
            { file: 'train.csv', n_rows: 200000, n_cols: 18, key_candidates: ['id'] },
            { file: 'test.csv', n_rows: 50000, n_cols: 18, key_candidates: ['id'] },
          ],
          same_schema_groups: [['test.csv', 'train.csv']],
          note: 'One file feeds the analysis; nothing was assembled. This report describes the bundle structure only.',
        },
      },
    },
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

  const traces: AgentTrace[] = [
    {
      agent_name: 'dag_expert',
      timestamp: stamp(140),
      action: 'step_1 · get_dag_adjustment_set',
      reasoning:
        'Treatment is `treatment`, outcome `re78`. Block the back-door paths via baseline covariates: age and education confound both training uptake and later earnings.',
      duration_ms: 1840,
      inputs: { treatment: 'treatment', outcome: 're78' },
      outputs: { status: 'ok', output: 'adjustment set: {age, educ, married}' },
      tools_called: ['get_dag_adjustment_set'],
      token_usage: { input_tokens: 220, output_tokens: 120 },
    },
    {
      agent_name: 'dag_expert',
      timestamp: stamp(135),
      action: 'step_2 · check_identifiability',
      reasoning:
        'With {age, educ, married} the back-door criterion is satisfied and no collider is opened. The effect is identifiable by adjustment.',
      duration_ms: 920,
      inputs: { adjustment_set: ['age', 'educ', 'married'] },
      outputs: { status: 'ok', output: 'identifiable via back-door adjustment' },
      tools_called: ['check_identifiability'],
      token_usage: { input_tokens: 180, output_tokens: 90 },
    },
    {
      agent_name: 'effect_estimator',
      timestamp: stamp(70),
      action: 'step_1 · run_method',
      reasoning:
        'Binary treatment with good overlap. Start with IPW, then a DML cross-fit as a doubly-robust check.',
      duration_ms: 4100,
      inputs: { method: 'ipw' },
      outputs: { status: 'ok', output: 'ATE 1.04 (95% CI 0.22, 1.86)' },
      tools_called: ['run_method'],
      token_usage: { input_tokens: 260, output_tokens: 140 },
    },
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

  return { job, events, traces };
}
