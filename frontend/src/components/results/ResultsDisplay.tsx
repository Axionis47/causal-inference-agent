import { AnalysisResults } from '../../services/api';
import { BarChart3, GitBranch, Shield, Lightbulb } from 'lucide-react';

interface ResultsDisplayProps {
  results: AnalysisResults;
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  return (
    <div className="space-y-6">
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
              {results.treatment_effects.map((effect, index) => (
                <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 text-sm font-medium text-gray-900">{effect.method}</td>
                  <td className="py-3 px-4 text-sm text-gray-600">{effect.estimand}</td>
                  <td className="py-3 px-4 text-sm text-right font-mono">
                    {effect.estimate.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-sm text-right font-mono text-gray-600">
                    {effect.std_error.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-sm text-center font-mono text-gray-600">
                    [{effect.ci_lower.toFixed(4)}, {effect.ci_upper.toFixed(4)}]
                  </td>
                  <td className="py-3 px-4 text-sm text-right font-mono">
                    {effect.p_value ? effect.p_value.toFixed(4) : 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Visual comparison */}
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Estimate Comparison</h3>
          <div className="space-y-2">
            {results.treatment_effects.map((effect, index) => {
              const maxAbs = Math.max(
                ...results.treatment_effects.map((e) =>
                  Math.max(Math.abs(e.ci_lower), Math.abs(e.ci_upper))
                )
              );
              const scale = 100 / maxAbs;
              const left = 50 + effect.ci_lower * scale * 0.5;
              const width = (effect.ci_upper - effect.ci_lower) * scale * 0.5;
              const center = 50 + effect.estimate * scale * 0.5;

              return (
                <div key={index} className="flex items-center">
                  <div className="w-40 text-xs text-gray-600 truncate pr-2">{effect.method}</div>
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
            {results.causal_graph.nodes.map((node) => (
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
          {results.causal_graph.edges.length > 0 && (
            <div className="mb-4">
              <span className="text-sm font-medium text-gray-700 block mb-2">
                Discovered Edges ({results.causal_graph.edges.length}):
              </span>
              <div className="flex flex-wrap gap-2">
                {results.causal_graph.edges.map((edge, idx) => (
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
      {results.sensitivity_analysis.length > 0 && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Shield className="w-5 h-5 text-primary-600" />
            <h2 className="text-lg font-semibold text-gray-900">Sensitivity Analysis</h2>
          </div>
          <div className="space-y-4">
            {results.sensitivity_analysis.map((sens, index) => (
              <div key={index} className="border-b border-gray-100 pb-4 last:border-0 last:pb-0">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{sens.method}</span>
                  <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                    {sens.robustness_value.toFixed(2)}
                  </span>
                </div>
                <p className="text-sm text-gray-600">{sens.interpretation}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {results.recommendations.length > 0 && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <Lightbulb className="w-5 h-5 text-primary-600" />
            <h2 className="text-lg font-semibold text-gray-900">Recommendations</h2>
          </div>
          <ul className="space-y-2">
            {results.recommendations.map((rec, index) => (
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
