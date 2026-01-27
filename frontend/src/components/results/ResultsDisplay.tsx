import { memo, useMemo } from 'react';
import { AnalysisResults } from '../../services/api';
import {
  BarChart3,
  GitBranch,
  Shield,
  Lightbulb,
  TrendingUp,
  TrendingDown,
  Minus,
  HelpCircle,
  CheckCircle,
  AlertTriangle,
  Database,
  Users,
} from 'lucide-react';

interface ResultsDisplayProps {
  results: AnalysisResults;
}

function ResultsDisplay({ results }: ResultsDisplayProps) {
  // Safely access arrays with fallbacks - memoize to maintain stable references
  const treatmentEffects = useMemo(
    () => results.treatment_effects ?? [],
    [results.treatment_effects]
  );
  const sensitivityAnalysis = useMemo(
    () => results.sensitivity_analysis ?? [],
    [results.sensitivity_analysis]
  );
  const recommendations = useMemo(
    () => results.recommendations ?? [],
    [results.recommendations]
  );

  // Memoize maxAbs calculation to avoid O(n²) complexity
  const maxAbs = useMemo(() => {
    if (treatmentEffects.length === 0) return 1;
    return Math.max(
      ...treatmentEffects.map((e) =>
        Math.max(Math.abs(e.ci_lower ?? 0), Math.abs(e.ci_upper ?? 0))
      )
    );
  }, [treatmentEffects]);

  // Get effect direction icon
  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'positive':
        return <TrendingUp className="w-5 h-5 text-green-600" />;
      case 'negative':
        return <TrendingDown className="w-5 h-5 text-red-600" />;
      case 'null':
        return <Minus className="w-5 h-5 text-gray-500" />;
      default:
        return <HelpCircle className="w-5 h-5 text-yellow-500" />;
    }
  };

  // Get confidence badge style
  const getConfidenceBadge = (level: string) => {
    switch (level) {
      case 'high':
        return { bg: 'bg-green-100', text: 'text-green-800', label: 'High Confidence' };
      case 'medium':
        return { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'Medium Confidence' };
      default:
        return { bg: 'bg-red-100', text: 'text-red-800', label: 'Low Confidence' };
    }
  };

  // Get consensus strength badge
  const getConsensusBadge = (strength: string) => {
    switch (strength) {
      case 'strong':
        return { bg: 'bg-green-100', text: 'text-green-700', icon: CheckCircle };
      case 'moderate':
        return { bg: 'bg-yellow-100', text: 'text-yellow-700', icon: AlertTriangle };
      default:
        return { bg: 'bg-red-100', text: 'text-red-700', icon: AlertTriangle };
    }
  };

  return (
    <div className="space-y-6">
      {/* Executive Summary - The Key Finding */}
      {results.executive_summary && (
        <div className="card bg-gradient-to-r from-primary-50 to-white border-l-4 border-primary-500">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              {getDirectionIcon(results.executive_summary.effect_direction)}
              <h2 className="text-lg font-semibold text-gray-900">Key Finding</h2>
            </div>
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                getConfidenceBadge(results.executive_summary.confidence_level).bg
              } ${getConfidenceBadge(results.executive_summary.confidence_level).text}`}
            >
              {getConfidenceBadge(results.executive_summary.confidence_level).label}
            </span>
          </div>

          <p className="text-lg text-gray-800 font-medium mb-4">
            {results.executive_summary.headline}
          </p>

          {results.executive_summary.key_findings.length > 0 && (
            <ul className="space-y-2">
              {results.executive_summary.key_findings.map((finding, idx) => (
                <li key={idx} className="flex items-start space-x-2 text-sm text-gray-600">
                  <CheckCircle className="w-4 h-4 text-primary-500 flex-shrink-0 mt-0.5" />
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Data Context & Method Consensus Row */}
      {(results.data_context || results.method_consensus) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Data Context */}
          {results.data_context && (
            <div className="card">
              <div className="flex items-center space-x-2 mb-3">
                <Database className="w-5 h-5 text-primary-600" />
                <h3 className="font-semibold text-gray-900">Data Context</h3>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-500">Sample Size</span>
                  <p className="font-medium text-gray-900">
                    {results.data_context.n_samples.toLocaleString()}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500">Features</span>
                  <p className="font-medium text-gray-900">{results.data_context.n_features}</p>
                </div>
                {results.data_context.n_treated != null && results.data_context.n_control != null && (
                  <>
                    <div>
                      <span className="text-gray-500">Treated</span>
                      <p className="font-medium text-gray-900">
                        {results.data_context.n_treated.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-500">Control</span>
                      <p className="font-medium text-gray-900">
                        {results.data_context.n_control.toLocaleString()}
                      </p>
                    </div>
                  </>
                )}
                {results.data_context.missing_data_pct > 0 && (
                  <div className="col-span-2">
                    <span className="text-gray-500">Missing Data</span>
                    <p className="font-medium text-yellow-600">
                      {results.data_context.missing_data_pct.toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>
              {results.data_context.data_quality_issues.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <span className="text-xs text-gray-500 uppercase tracking-wide">Quality Issues</span>
                  <ul className="mt-1 space-y-1">
                    {results.data_context.data_quality_issues.map((issue, idx) => (
                      <li key={idx} className="text-sm text-yellow-600 flex items-start space-x-1">
                        <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" />
                        <span>{issue}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Method Consensus */}
          {results.method_consensus && (
            <div className="card">
              <div className="flex items-center space-x-2 mb-3">
                <Users className="w-5 h-5 text-primary-600" />
                <h3 className="font-semibold text-gray-900">Method Consensus</h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-500">Agreement</span>
                  <div className="flex items-center space-x-2">
                    {(() => {
                      const badge = getConsensusBadge(results.method_consensus!.consensus_strength);
                      const Icon = badge.icon;
                      return (
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${badge.bg} ${badge.text} flex items-center space-x-1`}>
                          <Icon className="w-3 h-3" />
                          <span className="capitalize">{results.method_consensus!.consensus_strength}</span>
                        </span>
                      );
                    })()}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-500">Methods Used</span>
                    <p className="font-medium text-gray-900">{results.method_consensus.n_methods}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">Direction Agreement</span>
                    <p className="font-medium text-gray-900">
                      {Math.round(results.method_consensus.direction_agreement * 100)}%
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">Median Estimate</span>
                    <p className="font-mono font-medium text-gray-900">
                      {results.method_consensus.median_estimate.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">Estimate Range</span>
                    <p className="font-mono text-xs text-gray-600">
                      {results.method_consensus.estimate_range ? (
                        <>
                          [{results.method_consensus.estimate_range[0]?.toFixed(3) ?? 'N/A'},{' '}
                          {results.method_consensus.estimate_range[1]?.toFixed(3) ?? 'N/A'}]
                        </>
                      ) : (
                        'N/A'
                      )}
                    </p>
                  </div>
                </div>
                {results.method_consensus.all_significant && (
                  <div className="pt-2 border-t border-gray-100">
                    <span className="text-sm text-green-600 flex items-center space-x-1">
                      <CheckCircle className="w-4 h-4" />
                      <span>All methods show statistical significance</span>
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Treatment Effects */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-gray-900">Treatment Effect Estimates</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Method</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">Estimand</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-gray-500">Estimate</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-gray-500">Std. Error</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-gray-500">95% CI</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-gray-500">p-value</th>
              </tr>
            </thead>
            <tbody>
              {treatmentEffects.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-gray-500">
                    No treatment effect estimates available
                  </td>
                </tr>
              ) : (
                treatmentEffects.map((effect, index) => {
                  const isSignificant = effect.p_value != null && effect.p_value < 0.05;
                  return (
                    <tr
                      key={index}
                      className={`border-b border-gray-100 hover:bg-gray-50 ${
                        isSignificant ? 'bg-green-50/50' : ''
                      }`}
                    >
                      <td className="py-3 px-4 text-sm font-medium text-gray-900">
                        <div className="flex items-center space-x-2">
                          <span>{effect.method}</span>
                          {isSignificant && (
                            <span className="px-1.5 py-0.5 text-xs bg-green-100 text-green-700 rounded">
                              Sig.
                            </span>
                          )}
                        </div>
                        {effect.assumptions_tested && effect.assumptions_tested.length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {effect.assumptions_tested.map((assumption, aIdx) => (
                              <span
                                key={aIdx}
                                className="text-xs text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded"
                              >
                                {assumption}
                              </span>
                            ))}
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600">{effect.estimand}</td>
                      <td className={`py-3 px-4 text-sm text-right font-mono ${isSignificant ? 'font-semibold' : ''}`}>
                        {effect.estimate?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="py-3 px-4 text-sm text-right font-mono text-gray-600">
                        {effect.std_error?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="py-3 px-4 text-sm text-center font-mono text-gray-600">
                        [{effect.ci_lower?.toFixed(4) ?? '?'}, {effect.ci_upper?.toFixed(4) ?? '?'}]
                      </td>
                      <td className={`py-3 px-4 text-sm text-right font-mono ${
                        isSignificant ? 'text-green-600 font-semibold' : ''
                      }`}>
                        {effect.p_value != null ? (
                          effect.p_value < 0.001 ? '<0.001' : effect.p_value.toFixed(4)
                        ) : 'N/A'}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Visual comparison */}
        {treatmentEffects.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Estimate Comparison</h3>
          <div className="space-y-2" role="list" aria-label="Treatment effect comparison">
            {treatmentEffects.map((effect, index) => {
              const scale = 100 / maxAbs;
              const left = 50 + effect.ci_lower * scale * 0.5;
              const width = (effect.ci_upper - effect.ci_lower) * scale * 0.5;
              const center = 50 + effect.estimate * scale * 0.5;

              return (
                <div key={`effect-${effect.method}-${index}`} className="flex items-center" role="listitem">
                  <div className="w-32 sm:w-40 text-xs text-gray-600 truncate pr-2">{effect.method}</div>
                  <div className="flex-1 relative h-6 bg-gray-100 rounded">
                    {/* Zero line */}
                    <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-300" />
                    {/* CI bar */}
                    <div
                      className="absolute top-1 bottom-1 bg-primary-200 rounded"
                      style={{ left: `${left}%`, width: `${width}%` }}
                    />
                    {/* Point estimate */}
                    <div
                      className="absolute top-1/2 w-2 h-2 -mt-1 bg-primary-600 rounded-full"
                      style={{ left: `${center}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        )}
      </div>

      {/* Causal Graph */}
      {results.causal_graph && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <GitBranch className="w-5 h-5 text-primary-600" />
            <h2 className="text-lg font-semibold text-gray-900">Causal Graph</h2>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Discovered using: {results.causal_graph.discovery_method}
          </p>

          {/* Nodes */}
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="text-sm font-medium text-gray-700">Nodes:</span>
            {(results.causal_graph.nodes ?? []).map((node) => (
              <span
                key={node}
                className={`px-2 py-1 text-xs rounded-full ${
                  node === results.treatment_variable
                    ? 'bg-green-100 text-green-700'
                    : node === results.outcome_variable
                    ? 'bg-red-100 text-red-700'
                    : 'bg-gray-100 text-gray-700'
                }`}
              >
                {node}
              </span>
            ))}
          </div>

          {/* Edges */}
          {(results.causal_graph.edges?.length ?? 0) > 0 && (
            <div className="mb-4">
              <span className="text-sm font-medium text-gray-700 block mb-2">
                Discovered Edges ({results.causal_graph.edges?.length ?? 0}):
              </span>
              <div className="flex flex-wrap gap-2">
                {(results.causal_graph.edges ?? []).map((edge, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center px-3 py-1 text-xs bg-blue-50 text-blue-700 rounded-full"
                  >
                    <span className="font-medium">{edge.source}</span>
                    <span className="mx-1">→</span>
                    <span className="font-medium">{edge.target}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* LLM Interpretation */}
          {results.causal_graph.interpretation && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium text-gray-700 block mb-2">
                Interpretation:
              </span>
              <p className="text-sm text-gray-600 whitespace-pre-wrap">
                {results.causal_graph.interpretation}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Sensitivity Analysis */}
      {sensitivityAnalysis.length > 0 && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Shield className="w-5 h-5 text-primary-600" />
            <h2 className="text-lg font-semibold text-gray-900">Sensitivity Analysis</h2>
          </div>
          <div className="space-y-4">
            {sensitivityAnalysis.map((sens, index) => (
              <div key={index} className="border-b border-gray-100 pb-4 last:border-0 last:pb-0">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{sens.method}</span>
                  <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                    {sens.robustness_value?.toFixed(2) ?? 'N/A'}
                  </span>
                </div>
                <p className="text-sm text-gray-600">{sens.interpretation}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Lightbulb className="w-5 h-5 text-primary-600" />
            <h2 className="text-lg font-semibold text-gray-900">Recommendations</h2>
          </div>
          <ul className="space-y-2">
            {recommendations.map((rec, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="w-5 h-5 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                  {index + 1}
                </span>
                <span className="text-gray-700">{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default memo(ResultsDisplay);
