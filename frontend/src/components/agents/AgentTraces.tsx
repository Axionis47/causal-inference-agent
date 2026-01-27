import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, ChevronRight, Clock, Bot } from 'lucide-react';
import { getTraces, AgentTrace } from '../../services/api';

interface AgentTracesProps {
  jobId: string;
}

export default function AgentTraces({ jobId }: AgentTracesProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const tracesQuery = useQuery({
    queryKey: ['traces', jobId],
    queryFn: () => getTraces(jobId),
    enabled: isExpanded,
  });

  return (
    <div className="card">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center space-x-2">
          <Bot className="w-5 h-5 text-primary-600" />
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
              <div className="w-6 h-6 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : tracesQuery.data && tracesQuery.data.length > 0 ? (
            <div className="space-y-4">
              {tracesQuery.data.map((trace, index) => (
                <TraceCard
                  key={`${trace.agent_name}-${trace.timestamp}-${trace.action}-${index}`}
                  trace={trace}
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No traces available</p>
          )}
        </div>
      )}
    </div>
  );
}

function TraceCard({ trace }: { trace: AgentTrace }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const agentColors: Record<string, string> = {
    orchestrator: 'bg-purple-100 text-purple-700',
    data_profiler: 'bg-blue-100 text-blue-700',
    causal_discovery: 'bg-green-100 text-green-700',
    effect_estimator: 'bg-yellow-100 text-yellow-700',
    sensitivity_analyst: 'bg-orange-100 text-orange-700',
    critique: 'bg-red-100 text-red-700',
    notebook_generator: 'bg-indigo-100 text-indigo-700',
  };

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-50 text-left"
      >
        <div className="flex items-center space-x-3">
          <span
            className={`px-2 py-1 text-xs font-medium rounded-full ${
              agentColors[trace.agent_name] || 'bg-gray-100 text-gray-700'
            }`}
          >
            {trace.agent_name.replace('_', ' ')}
          </span>
          <span className="text-sm font-medium text-gray-900">{trace.action}</span>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center text-xs text-gray-500">
            <Clock className="w-3 h-3 mr-1" />
            {trace.duration_ms}ms
          </div>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {isExpanded && trace.reasoning && (
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          <p className="text-sm font-medium text-gray-700 mb-2">Reasoning:</p>
          <p className="text-sm text-gray-600 whitespace-pre-wrap">{trace.reasoning}</p>
        </div>
      )}
    </div>
  );
}
