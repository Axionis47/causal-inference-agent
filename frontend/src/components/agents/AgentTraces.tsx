import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, ChevronRight, Clock, Bot, Filter, Zap } from 'lucide-react';
import { getTraces, AgentTrace } from '../../services/api';

interface AgentTracesProps {
  jobId: string;
}

export default function AgentTraces({ jobId }: AgentTracesProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [agentFilter, setAgentFilter] = useState<string>('all');

  const tracesQuery = useQuery({
    queryKey: ['traces', jobId],
    queryFn: () => getTraces(jobId),
    enabled: isExpanded,
  });

  const uniqueAgents = useMemo(() => {
    if (!tracesQuery.data) return [];
    const agents = new Set(tracesQuery.data.map((t) => t.agent_name));
    return Array.from(agents).sort();
  }, [tracesQuery.data]);

  const filteredTraces = useMemo(() => {
    if (!tracesQuery.data) return [];
    if (agentFilter === 'all') return tracesQuery.data;
    return tracesQuery.data.filter((t) => t.agent_name === agentFilter);
  }, [tracesQuery.data, agentFilter]);

  const totals = useMemo(() => {
    const traces = tracesQuery.data || [];
    return {
      count: traces.length,
      duration: traces.reduce((s, t) => s + t.duration_ms, 0),
      inputTokens: traces.reduce((s, t) => s + (t.token_usage?.input_tokens || 0), 0),
      outputTokens: traces.reduce((s, t) => s + (t.token_usage?.output_tokens || 0), 0),
    };
  }, [tracesQuery.data]);

  return (
    <div className="card">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center space-x-2">
          <Bot className="w-5 h-5 text-gray-900" />
          <h2 className="text-lg font-semibold text-gray-900">Agent Reasoning Traces</h2>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-4">
          {tracesQuery.isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="w-6 h-6 border-2 border-gray-900 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : tracesQuery.data && tracesQuery.data.length > 0 ? (
            <>
              {/* Summary bar */}
              <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500 mb-3 bg-gray-50 rounded-xl px-3 py-2">
                <span className="font-medium">{totals.count} traces</span>
                <span className="flex items-center">
                  <Clock className="w-3 h-3 mr-1" />
                  {(totals.duration / 1000).toFixed(1)}s total
                </span>
                {(totals.inputTokens > 0 || totals.outputTokens > 0) && (
                  <span className="flex items-center">
                    <Zap className="w-3 h-3 mr-1" />
                    {totals.inputTokens.toLocaleString()} in / {totals.outputTokens.toLocaleString()} out tokens
                  </span>
                )}
              </div>

              {/* Agent filter */}
              {uniqueAgents.length > 1 && (
                <div className="flex items-center space-x-2 mb-3">
                  <Filter className="w-3.5 h-3.5 text-gray-400" />
                  <select
                    value={agentFilter}
                    onChange={(e) => setAgentFilter(e.target.value)}
                    className="text-xs border border-gray-200 rounded-lg px-2 py-1 bg-white focus:ring-2 focus:ring-gray-900/20 focus:border-gray-400"
                  >
                    <option value="all">All Agents ({tracesQuery.data.length})</option>
                    {uniqueAgents.map((a) => (
                      <option key={a} value={a}>
                        {a.replace(/_/g, ' ')} ({tracesQuery.data!.filter((t) => t.agent_name === a).length})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div className="space-y-3">
                {filteredTraces.map((trace, index) => (
                  <TraceCard
                    key={`${trace.agent_name}-${trace.timestamp}-${trace.action}-${index}`}
                    trace={trace}
                  />
                ))}
              </div>
            </>
          ) : (
            <p className="text-gray-500 text-center py-4">No traces available</p>
          )}
        </div>
      )}
    </div>
  );
}

const agentBadgeColors: Record<string, string> = {
  orchestrator: 'bg-purple-100 text-purple-700',
  data_profiler: 'bg-blue-100 text-blue-700',
  causal_discovery: 'bg-green-100 text-green-700',
  effect_estimator: 'bg-yellow-100 text-yellow-700',
  sensitivity_analyst: 'bg-orange-100 text-orange-700',
  critique: 'bg-red-100 text-red-700',
  notebook_generator: 'bg-indigo-100 text-indigo-700',
  dag_expert: 'bg-teal-100 text-teal-700',
  confounder_discovery: 'bg-cyan-100 text-cyan-700',
  ps_diagnostics: 'bg-lime-100 text-lime-700',
  data_repair: 'bg-rose-100 text-rose-700',
  domain_knowledge: 'bg-violet-100 text-violet-700',
};

const agentBorderColors: Record<string, string> = {
  orchestrator: 'border-l-purple-400',
  data_profiler: 'border-l-blue-400',
  causal_discovery: 'border-l-green-400',
  effect_estimator: 'border-l-yellow-400',
  sensitivity_analyst: 'border-l-orange-400',
  critique: 'border-l-red-400',
  notebook_generator: 'border-l-indigo-400',
  dag_expert: 'border-l-teal-400',
  confounder_discovery: 'border-l-cyan-400',
  ps_diagnostics: 'border-l-lime-400',
  data_repair: 'border-l-rose-400',
  domain_knowledge: 'border-l-violet-400',
};

function TraceCard({ trace }: { trace: AgentTrace }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const borderColor = agentBorderColors[trace.agent_name] || 'border-l-gray-400';
  const badgeColor = agentBadgeColors[trace.agent_name] || 'bg-gray-100 text-gray-700';
  const hasTokens = (trace.token_usage?.input_tokens || 0) > 0 || (trace.token_usage?.output_tokens || 0) > 0;

  return (
    <div className={`border-l-4 ${borderColor} border border-gray-100 rounded-lg overflow-hidden shadow-sm`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-50 text-left transition-colors"
      >
        <div className="flex items-center space-x-3 min-w-0">
          <span className={`px-2 py-1 text-xs font-semibold rounded-full whitespace-nowrap ${badgeColor}`}>
            {trace.agent_name.replace(/_/g, ' ')}
          </span>
          <span className="text-sm font-medium text-gray-900 truncate">{trace.action}</span>
        </div>
        <div className="flex items-center space-x-3 flex-shrink-0">
          {hasTokens && (
            <span className="text-xs text-gray-400">
              {(trace.token_usage.input_tokens || 0) + (trace.token_usage.output_tokens || 0)} tok
            </span>
          )}
          <div className="flex items-center text-xs text-gray-500">
            <Clock className="w-3 h-3 mr-1" />
            {trace.duration_ms >= 1000
              ? `${(trace.duration_ms / 1000).toFixed(1)}s`
              : `${trace.duration_ms}ms`}
          </div>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-gray-100 p-4 bg-gray-50 space-y-3">
          {/* Reasoning */}
          {trace.reasoning && (
            <div>
              <p className="text-sm font-semibold text-gray-700 mb-1">Reasoning:</p>
              <p className="text-sm text-gray-600 whitespace-pre-wrap">{trace.reasoning}</p>
            </div>
          )}

          {/* Tools Called */}
          {trace.tools_called?.length > 0 && (
            <div>
              <p className="text-sm font-semibold text-gray-700 mb-1">Tools Called:</p>
              <div className="flex flex-wrap gap-1">
                {trace.tools_called.map((tool, i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs font-medium rounded-md border border-blue-100"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Inputs */}
          {trace.inputs && Object.keys(trace.inputs).length > 0 && (
            <CollapsibleJson label="Inputs" data={trace.inputs} />
          )}

          {/* Outputs */}
          {trace.outputs && Object.keys(trace.outputs).length > 0 && (
            <CollapsibleJson label="Outputs" data={trace.outputs} />
          )}

          {/* Token Usage & Timestamp */}
          <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400 pt-1 border-t border-gray-200">
            {hasTokens && (
              <span>
                Tokens: {trace.token_usage.input_tokens || 0} in / {trace.token_usage.output_tokens || 0} out
              </span>
            )}
            {trace.timestamp && (
              <span>{new Date(trace.timestamp).toLocaleTimeString()}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function CollapsibleJson({ label, data }: { label: string; data: Record<string, unknown> }) {
  const [open, setOpen] = useState(false);

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="text-sm font-semibold text-gray-700 flex items-center space-x-1 hover:text-gray-900 transition-colors"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <span>{label}</span>
        <span className="text-xs font-normal text-gray-400">({Object.keys(data).length} fields)</span>
      </button>
      {open && (
        <pre className="mt-1 text-xs text-gray-600 bg-white border border-gray-200 rounded-lg p-3 overflow-x-auto max-h-48">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}
