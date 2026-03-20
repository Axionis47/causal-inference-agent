import { memo, useMemo, useEffect, useState } from 'react';
import { AnalysisResults } from '../../services/api';
import CausalGraphView from './CausalGraphView';
import ForestPlot from './ForestPlot';
import Tooltip from '../common/Tooltip';
import glossary from '../../utils/glossary';
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
  // Fade-in animation for hero card
  const [heroVisible, setHeroVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setHeroVisible(true), 50);
    return () => clearTimeout(t);
  }, []);

  // Helper to wrap a term in a Tooltip if it exists in the glossary
  const tip = (term: string, label?: string) =>
    glossary[term] ? <Tooltip term={term}>{label ?? term}</Tooltip> : <>{label ?? term}</>;

  // Derive hero card content
  const heroText = results.narrative_summary || results.executive_summary?.headline || '';
  const heroDirection = results.executive_summary?.effect_direction ?? 'null';
  const heroConfidence = results.executive_summary?.confidence_level ?? 'low';

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
      {/* Narrative Summary Hero Card */}
      {heroText && (
        <div
          className="w-full rounded-xl p-6 md:p-8"
          style={{
            backgroundColor: '#f0f4f8',
            opacity: heroVisible ? 1 : 0,
            transform: heroVisible ? 'translateY(0)' : 'translateY(8px)',
            transition: 'opacity 0.5s ease-out, transform 0.5s ease-out',
          }}
        >
          <div className="flex items-start gap-4">
            {/* Direction indicator */}
            <span className="flex-shrink-0 mt-1 text-2xl" aria-hidden="true">
              {heroDirection === 'positive' && <TrendingUp className="w-7 h-7 text-green-600" />}
              {heroDirection === 'negative' && <TrendingDown className="w-7 h-7 text-red-600" />}
              {(heroDirection === 'null' || heroDirection === 'mixed') && (
                <Minus className="w-7 h-7 text-gray-400" />
              )}
            </span>

            <div className="flex-1 min-w-0">
              <p
                className="text-gray-800 font-medium"
                style={{
                  fontSize: '1.2rem',
                  lineHeight: '1.75',
                  fontFamily: 'Georgia, "Times New Roman", serif',
                }}
              >
                {heroText}
              </p>
            </div>

            {/* Confidence badge */}
            <span
              className={`flex-shrink-0 px-3 py-1 rounded-full text-sm font-medium ${
                heroConfidence === 'high'
                  ? 'bg-green-100 text-green-800'
                  : heroConfidence === 'medium'
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
              }`}
            >
              {heroConfidence === 'high'
                ? 'High Confidence'
                : heroConfidence === 'medium'
                ? 'Moderate'
                : 'Low'}
            </span>
          </div>
        </div>
      )}

      {/* Executive Summary - The Key Finding */}
      {results.executive_summary && (
        <div className="card border-l-4 border-gray-900">
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
                  <CheckCircle className="w-4 h-4 text-gray-400 flex-shrink-0 mt-0.5" />
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
                <Database className="w-5 h-5 text-gray-900" />
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
                <Users className="w-5 h-5 text-gray-900" />
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
          <BarChart3 className="w-5 h-5 text-gray-900" />
          <h2 className="text-lg font-semibold text-gray-900">Treatment Effect Estimates</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50/80">
                <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wider">Method</th>
                <th className="text-left py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wider">Estimand</th>
                <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wider">Estimate</th>
                <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wider">Std. Error</th>
                <th className="text-center py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wider">{tip('95% CI')}</th>
                <th className="text-right py-3 px-5 text-xs font-semibold text-gray-500 uppercase tracking-wider">{tip('p-value')}</th>
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
                      className={`border-b border-gray-100 hover:bg-gray-100/50 ${
                        isSignificant ? 'bg-green-50/50' : index % 2 === 1 ? 'bg-gray-50/50' : ''
                      }`}
                    >
                      <td className="py-3.5 px-5 text-sm font-medium text-gray-900">
                        <div className="flex items-center space-x-2">
                          <span>{tip(effect.method)}</span>
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
                      <td className="py-3.5 px-5 text-sm text-gray-600">{tip(effect.estimand)}</td>
                      <td className={`py-3.5 px-5 text-sm text-right font-mono ${isSignificant ? 'font-semibold' : ''}`}>
                        {effect.estimate?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="py-3.5 px-5 text-sm text-right font-mono text-gray-600">
                        {effect.std_error?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="py-3.5 px-5 text-sm text-center font-mono text-gray-600">
                        [{effect.ci_lower?.toFixed(4) ?? '?'}, {effect.ci_upper?.toFixed(4) ?? '?'}]
                      </td>
                      <td className={`py-3.5 px-5 text-sm text-right font-mono ${
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

        {/* Forest Plot */}
        {treatmentEffects.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Forest Plot</h3>
          <ForestPlot effects={treatmentEffects} />
        </div>
        )}
      </div>

      {/* Causal Graph */}
      {results.causal_graph && (
        <div className="card">
          <div className="flex items-center space-x-2 mb-4">
            <GitBranch className="w-5 h-5 text-gray-900" />
            <h2 className="text-lg font-semibold text-gray-900">Causal Graph</h2>
          </div>
          <p className="text-sm text-gray-600 mb-4">
            Discovered using: {results.causal_graph.discovery_method}
          </p>

          {/* Interactive SVG Graph */}
          <CausalGraphView
            graph={results.causal_graph}
            treatmentVariable={results.treatment_variable}
            outcomeVariable={results.outcome_variable}
          />

          {/* Legend */}
          <div className="flex flex-wrap gap-4 mt-4 justify-center text-xs text-gray-600">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full bg-[#22c55e]" /> Treatment
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full bg-[#ef4444]" /> Outcome
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full bg-[#6b7280]" /> Other
            </span>
          </div>

          {/* LLM Interpretation */}
          {results.causal_graph.interpretation && (
            <div className="mt-4 p-4 bg-gray-50 rounded-xl">
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
            <Shield className="w-5 h-5 text-gray-900" />
            <h2 className="text-lg font-semibold text-gray-900">Sensitivity Analysis</h2>
          </div>
          <div className="space-y-4">
            {sensitivityAnalysis.map((sens, index) => (
              <div key={index} className="border-b border-gray-100 pb-4 last:border-0 last:pb-0">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{tip(sens.method)}</span>
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
            <Lightbulb className="w-5 h-5 text-gray-900" />
            <h2 className="text-lg font-semibold text-gray-900">Recommendations</h2>
          </div>
          <ul className="space-y-2">
            {recommendations.map((rec, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="w-5 h-5 rounded-full bg-gray-900 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
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
