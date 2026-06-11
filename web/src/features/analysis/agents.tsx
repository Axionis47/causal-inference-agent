import { useState } from "react";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";
import { ErrorMessage, Loading } from "@/components/feedback";
import { RecordDetails } from "./inspector";

type Analysis = components["schemas"]["AnalysisView"];
type Agent = components["schemas"]["AgentSummary"];

const labels: Record<string, string> = {
  pending: "Waiting",
  running: "In progress",
  accepted: "Complete",
  awaiting_approval: "Ready for approval",
  blocked: "Stopped",
  exhausted: "Allowance reached",
  failed: "Interrupted",
};

export function AgentProgress({ data }: { data: Analysis }) {
  if (!data.agents?.length) return null;
  return (
    <section className="analysis-section" aria-label="Analysis progress">
      <h3 className="eyebrow">Progress</h3>
      <p className="subtext mt-2">
        Using your applied context. No further context preparation is needed.
      </p>
      <ol className="mt-4 grid gap-3 sm:grid-cols-2">
        {data.agents.map((agent, index) => (
          <li
            key={agent.agent}
            className="border border-border p-3"
            aria-current={agent.agent === data.stage ? "step" : undefined}
          >
            <p>
              {String(index + 1).padStart(2, "0")} / {agent.title}
            </p>
            <p className="subtext mt-1">
              {labels[agent.status] ?? agent.status}
            </p>
            {agent.summary && <p className="subtext mt-2">{agent.summary}</p>}
          </li>
        ))}
      </ol>
    </section>
  );
}

function AgentRecord({ run, agent }: { run: string; agent: Agent }) {
  const query = api.useQuery(
    "get",
    "/api/analyses/{run_id}/agents/{agent_id}",
    {
      params: { path: { run_id: run, agent_id: agent.id! } },
    },
  );
  if (query.isPending) return <Loading>Opening saved agent work…</Loading>;
  if (query.error || !query.data) return <ErrorMessage error={query.error} />;
  return <RecordDetails value={query.data} />;
}

function AgentInspection({ run, agent }: { run: string; agent: Agent }) {
  const [open, setOpen] = useState(false);
  return (
    <details
      className="mt-3"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        {agent.title} / {agent.model_calls} of {agent.model_limit} calls
      </summary>
      <p className="subtext mt-2">
        {agent.submissions} handoff submissions ·{" "}
        {agent.input_tokens.toLocaleString()} input tokens ·{" "}
        {agent.output_tokens.toLocaleString()} output tokens ·{" "}
        {Math.round(agent.elapsed_seconds)} seconds in model calls
      </p>
      <p className="analysis-hash">
        Accepted handoff: {agent.handoff_id ?? "Not yet published"}
      </p>
      {open && agent.id && <AgentRecord run={run} agent={agent} />}
    </details>
  );
}

export function AgentInspector({ data }: { data: Analysis }) {
  return (
    <section className="mt-4" aria-label="Agent inspection">
      <h3 className="eyebrow">Agent sessions and accepted handoffs</h3>
      {data.agents?.map((agent) => (
        <AgentInspection key={agent.agent} run={data.id} agent={agent} />
      ))}
    </section>
  );
}
