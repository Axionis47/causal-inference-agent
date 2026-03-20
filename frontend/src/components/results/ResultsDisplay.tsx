import { memo, useMemo, useEffect, useState } from 'react';
import { AnalysisResults } from '../../services/api';
import CausalGraphView from './CausalGraphView';
import ForestPlot from './ForestPlot';
import Tooltip from '../common/Tooltip';
import glossary from '../../utils/glossary';

interface ResultsDisplayProps {
  results: AnalysisResults;
}

function ResultsDisplay({ results }: ResultsDisplayProps) {
  // Fade-in animation for abstract
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 50);
    return () => clearTimeout(t);
  }, []);

  // Helper to wrap a term in a Tooltip if it exists in the glossary
  const tip = (term: string, label?: string) =>
    glossary[term] ? <Tooltip term={term}>{label ?? term}</Tooltip> : <>{label ?? term}</>;

  // Derive abstract content
  const heroText = results.narrative_summary || results.executive_summary?.headline || '';

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
  const decisionLog = useMemo(
    () => results.decision_log ?? [],
    [results.decision_log]
  );

  // Determine if any method is significant
  const anySignificant = useMemo(
    () => treatmentEffects.some((e) => e.p_value != null && e.p_value < 0.05),
    [treatmentEffects]
  );

  // Figure counter
  let figureCount = 0;

  return (
    <article
      className="max-w-3xl mx-auto"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(8px)',
        transition: 'opacity 0.5s ease-out, transform 0.5s ease-out',
      }}
    >
      {/* ------------------------------------------------------------------ */}
      {/* 1. ABSTRACT                                                        */}
      {/* ------------------------------------------------------------------ */}
      {heroText && (
        <section className="pb-8 mb-0">
          <p
            className="font-serif text-ink-900 leading-relaxed"
            style={{ fontSize: '1.25rem', lineHeight: '1.9' }}
          >
            {heroText}
          </p>

          <p className="font-mono text-xs text-ink-500 mt-4 tracking-wide">
            {results.treatment_variable && (
              <>Treatment: {results.treatment_variable}</>
            )}
            {results.outcome_variable && (
              <> &nbsp;|&nbsp; Outcome: {results.outcome_variable}</>
            )}
            {treatmentEffects.length > 0 && (
              <> &nbsp;|&nbsp; Methods: {treatmentEffects.length}</>
            )}
            <> &nbsp;|&nbsp; Significance: {anySignificant ? 'yes' : 'no'}</>
          </p>

          <hr className="border-ink-200 mt-6" />
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 2. DATA                                                            */}
      {/* ------------------------------------------------------------------ */}
      {results.data_context && (
        <section className="border-t border-ink-200 pt-6 pb-8">
          <h2 className="font-serif text-xl font-bold text-ink-900 mb-4">Data</h2>

          <p className="font-sans text-ink-700 leading-relaxed">
            The dataset contains{' '}
            <span className="font-mono">{results.data_context.n_samples.toLocaleString()}</span>{' '}
            observations with{' '}
            <span className="font-mono">{results.data_context.n_features}</span> features.
            {results.data_context.n_treated != null && results.data_context.n_control != null && (
              <>
                {' '}The treatment group has{' '}
                <span className="font-mono">{results.data_context.n_treated.toLocaleString()}</span>{' '}
                units and the control group has{' '}
                <span className="font-mono">{results.data_context.n_control.toLocaleString()}</span>{' '}
                units.
              </>
            )}
            {results.data_context.missing_data_pct > 0 && (
              <>
                {' '}Missing data accounts for{' '}
                <span className="font-mono">{results.data_context.missing_data_pct.toFixed(1)}%</span>{' '}
                of all values.
              </>
            )}
          </p>

          {results.data_context.data_quality_issues.length > 0 && (
            <div className="mt-4">
              <p className="font-sans text-ink-700 text-sm mb-1">Data quality issues:</p>
              <ul className="list-disc list-inside text-sm text-ink-700 space-y-1 pl-1">
                {results.data_context.data_quality_issues.map((issue, idx) => (
                  <li key={idx}>{issue}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 3. METHODS AND RESULTS                                             */}
      {/* ------------------------------------------------------------------ */}
      <section className="border-t border-ink-200 pt-6 pb-8">
        <h2 className="font-serif text-xl font-bold text-ink-900 mb-6">Results</h2>

        {/* Table 1. Treatment Effect Estimates */}
        <p className="font-sans text-sm font-semibold text-ink-900 mb-2">
          Table 1. Treatment Effect Estimates
        </p>

        <div className="overflow-x-auto">
          <table className="journal-table">
            <thead>
              <tr>
                <th className="text-left">Method</th>
                <th className="text-left">{tip('estimand', 'Estimand')}</th>
                <th className="text-right">Estimate</th>
                <th className="text-right">Std. Error</th>
                <th className="text-center">{tip('95% CI', '95% CI')}</th>
                <th className="text-right">{tip('p-value', 'p-value')}</th>
              </tr>
            </thead>
            <tbody>
              {treatmentEffects.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-ink-500">
                    No treatment effect estimates available
                  </td>
                </tr>
              ) : (
                treatmentEffects.map((effect, index) => {
                  const isSignificant = effect.p_value != null && effect.p_value < 0.05;
                  return (
                    <tr
                      key={index}
                      style={isSignificant ? { borderLeft: '2px solid #15803d' } : undefined}
                    >
                      <td className="font-sans text-ink-900 font-medium">
                        {tip(effect.method)}
                      </td>
                      <td className="font-sans text-ink-700">
                        {tip(effect.estimand)}
                      </td>
                      <td className="text-right font-mono text-ink-900">
                        {effect.estimate?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="text-right font-mono text-ink-700">
                        {effect.std_error?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="text-center font-mono text-ink-700">
                        [{effect.ci_lower?.toFixed(4) ?? '?'}, {effect.ci_upper?.toFixed(4) ?? '?'}]
                      </td>
                      <td className={`text-right font-mono ${
                        isSignificant ? 'text-sig-yes font-semibold' : 'text-ink-700'
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
        {treatmentEffects.length > 0 && (() => {
          figureCount++;
          const figNum = figureCount;
          return (
            <div className="mt-8">
              <ForestPlot effects={treatmentEffects} />
              <p className="font-sans text-sm text-ink-500 italic mt-2">
                Figure {figNum}. Forest plot of treatment effect estimates with 95% confidence intervals.
              </p>
            </div>
          );
        })()}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* 4. CAUSAL STRUCTURE                                                */}
      {/* ------------------------------------------------------------------ */}
      {results.causal_graph && (() => {
        figureCount++;
        const figNum = figureCount;
        return (
          <section className="border-t border-ink-200 pt-6 pb-8">
            <h2 className="font-serif text-xl font-bold text-ink-900 mb-6">Causal Structure</h2>

            <CausalGraphView
              graph={results.causal_graph}
              treatmentVariable={results.treatment_variable}
              outcomeVariable={results.outcome_variable}
            />

            <p className="font-sans text-sm text-ink-500 italic mt-2">
              Figure {figNum}. Estimated causal directed acyclic graph.
            </p>

            <p className="font-sans text-ink-700 mt-4 leading-relaxed">
              The graph was discovered using the{' '}
              <span className="font-mono text-sm">{results.causal_graph.discovery_method}</span>{' '}
              algorithm.
            </p>

            {results.causal_graph.interpretation && (
              <p className="font-sans text-ink-700 mt-3 leading-relaxed">
                {results.causal_graph.interpretation}
              </p>
            )}
          </section>
        );
      })()}

      {/* ------------------------------------------------------------------ */}
      {/* 5. SENSITIVITY ANALYSIS                                            */}
      {/* ------------------------------------------------------------------ */}
      {sensitivityAnalysis.length > 0 && (
        <section className="border-t border-ink-200 pt-6 pb-8">
          <h2 className="font-serif text-xl font-bold text-ink-900 mb-6">Sensitivity Analysis</h2>

          <div className="overflow-x-auto">
            <table className="journal-table">
              <thead>
                <tr>
                  <th className="text-left">Method</th>
                  <th className="text-right">Robustness</th>
                  <th className="text-left">Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {sensitivityAnalysis.map((sens, index) => (
                  <tr key={index}>
                    <td className="font-sans text-ink-900 font-medium">
                      {tip(sens.method)}
                    </td>
                    <td className="text-right font-mono text-ink-900">
                      {sens.robustness_value?.toFixed(2) ?? 'N/A'}
                    </td>
                    <td className="font-sans text-ink-700">
                      {sens.interpretation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 6. METHODOLOGY DECISIONS                                           */}
      {/* ------------------------------------------------------------------ */}
      {decisionLog.length > 0 && (
        <section className="border-t border-ink-200 pt-6 pb-8">
          <h2 className="font-serif text-xl font-bold text-ink-900 mb-6">Methodology Decisions</h2>

          <div className="overflow-x-auto">
            <table className="journal-table">
              <thead>
                <tr>
                  <th className="text-left">Agent</th>
                  <th className="text-left">Decision</th>
                  <th className="text-left">Reason</th>
                </tr>
              </thead>
              <tbody>
                {decisionLog.map((d, index) => {
                  const isRejected = d.decision_type.includes('rejected') || d.decision_type.includes('failed');
                  const isQualityGate = d.decision_type === 'quality_gate';
                  const displayType = d.decision_type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
                  return (
                    <tr key={index}>
                      <td className="font-mono text-sm text-ink-900">{d.agent}</td>
                      <td className={`font-sans ${isRejected ? 'text-sig-no' : isQualityGate ? 'text-accent' : 'text-ink-900'}`}>
                        {displayType}: {d.choice}
                      </td>
                      <td className="font-sans text-ink-700">
                        {d.reason}
                        {d.alternatives && d.alternatives.length > 0 && (
                          <div className="mt-2 text-sm text-ink-500">
                            <span className="font-medium">Alternatives considered:</span>
                            <ul className="list-disc list-inside mt-1 space-y-0.5 pl-1">
                              {d.alternatives.map((alt, altIdx) => (
                                <li key={altIdx}>
                                  <span className="font-mono">{alt.option}</span> — {alt.reason}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* 7. DISCUSSION                                                      */}
      {/* ------------------------------------------------------------------ */}
      {(results.executive_summary || recommendations.length > 0 || results.method_consensus) && (
        <section className="border-t border-ink-200 pt-6 pb-8">
          <h2 className="font-serif text-xl font-bold text-ink-900 mb-6">Discussion</h2>

          {/* Key findings as paragraphs */}
          {results.executive_summary?.key_findings && results.executive_summary.key_findings.length > 0 && (
            <div className="space-y-3 mb-6">
              {results.executive_summary.key_findings.map((finding, idx) => (
                <p key={idx} className="font-sans text-ink-700 leading-relaxed">
                  {finding}
                </p>
              ))}
            </div>
          )}

          {/* Method consensus summary */}
          {results.method_consensus && (
            <p className="font-sans text-ink-700 leading-relaxed mb-6">
              Across{' '}
              <span className="font-mono">{results.method_consensus.n_methods}</span>{' '}
              estimation methods, direction agreement is{' '}
              <span className="font-mono">
                {Math.round(results.method_consensus.direction_agreement * 100)}%
              </span>{' '}
              with a median effect estimate of{' '}
              <span className="font-mono">
                {results.method_consensus.median_estimate.toFixed(4)}
              </span>{' '}
              (range:{' '}
              <span className="font-mono">
                {results.method_consensus.estimate_range
                  ? `${results.method_consensus.estimate_range[0]?.toFixed(3) ?? 'N/A'} to ${results.method_consensus.estimate_range[1]?.toFixed(3) ?? 'N/A'}`
                  : 'N/A'}
              </span>
              ). The consensus strength is{' '}
              <span className="font-mono">{results.method_consensus.consensus_strength}</span>.
              {results.method_consensus.all_significant &&
                ' All methods show statistical significance at the 0.05 level.'}
            </p>
          )}

          {/* Recommendations as numbered list */}
          {recommendations.length > 0 && (
            <ol className="list-decimal list-inside space-y-2">
              {recommendations.map((rec, index) => (
                <li key={index} className="font-sans text-ink-700 leading-relaxed">
                  {rec}
                </li>
              ))}
            </ol>
          )}
        </section>
      )}
    </article>
  );
}

export default memo(ResultsDisplay);
