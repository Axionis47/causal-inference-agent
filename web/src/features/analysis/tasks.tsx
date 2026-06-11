import { useId, useState } from "react";
import type { components } from "@/lib/api/schema";
import { CheckpointAttempt, EvidenceLinks, RecordDetails } from "./inspector";
import { taskProgress } from "./task-progress";

type Analysis = components["schemas"]["AnalysisView"];
type Task = components["schemas"]["AnalysisTaskView"];

function scopeRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function roleScopeLabel(task: Task, names: Readonly<Record<string, string>>) {
  const scope = scopeRecord(task.scope.role_scope);
  if (!scope) return "";
  const event = scopeRecord(scope.event);
  const treatments = Array.isArray(event?.treatment_ids)
    ? event.treatment_ids.filter((id): id is string => typeof id === "string")
    : [];
  const effectIds = Object.keys(scopeRecord(scope.effect_treatments) ?? {});
  const temporal = scopeRecord(event?.group_time);
  const adoption = scopeRecord(temporal?.adoption_period);
  return [
    typeof event?.kind === "string"
      ? `event kind: ${event.kind.replaceAll("_", " ")}`
      : "",
    treatments.length
      ? `event: ${treatments.map((id) => names[id] ?? id).join(", ")}`
      : "",
    typeof temporal?.cohort_variable_id === "string"
      ? `cohort variable: ${names[temporal.cohort_variable_id] ?? temporal.cohort_variable_id}`
      : "",
    typeof temporal?.cohort_value_id === "string"
      ? `cohort value: ${temporal.cohort_value_id}`
      : "",
    typeof temporal?.time_variable_id === "string"
      ? `time variable: ${names[temporal.time_variable_id] ?? temporal.time_variable_id}`
      : "",
    typeof adoption?.literal === "string" &&
    ["text", "number", "boolean"].includes(String(adoption.kind))
      ? `adoption: ${adoption.kind} ${adoption.kind === "text" ? JSON.stringify(adoption.literal) : adoption.literal}`
      : "",
    effectIds.length ? `effects: ${effectIds.join(", ")}` : "",
  ]
    .filter(Boolean)
    .join(" / ");
}

function offeredReferences(packet: Record<string, unknown>): string[] {
  if (!Array.isArray(packet.evidence_records)) return [];
  return packet.evidence_records.flatMap((record: unknown) => {
    if (!record || typeof record !== "object" || !("key" in record)) return [];
    return typeof record.key === "string" ? [record.key] : [];
  });
}

export function TaskInspector({
  data,
  inspect,
}: {
  data: Analysis;
  inspect: (key: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const selectorId = useId();
  const tasks = data.tasks ?? [];
  const active =
    tasks.find((task) => task.id === selected) ??
    tasks.find((task) => task.status === "running") ??
    tasks.at(-1);
  if (!active) return null;

  const receipts = new Map(
    [
      ...data.checkpoints.flatMap((checkpoint) => checkpoint.attempts),
      ...tasks.flatMap((task) => task.attempts),
    ].map((receipt) => [receipt.id, receipt]),
  );
  const progress = taskProgress(tasks);
  const providerOnly = active.attempt_details.filter(
    (detail) =>
      !active.attempts.some((receipt) => receipt.id === detail.receipt_id),
  );
  const offered = offeredReferences(active.packet);
  const columns =
    (
      data.context.content as
        components["schemas"]["ContextContent"] | undefined
    )?.columns ?? {};
  const labels = Object.fromEntries(
    Object.values(columns).map(({ identity }) => [
      `columns.${identity.column_id}`,
      identity.name,
    ]),
  );
  const names = Object.fromEntries(
    Object.values(columns).map(({ identity }) => [
      identity.column_id,
      identity.name,
    ]),
  );
  const sourceColumnCount = Object.keys(columns).length;
  const activeRoleScope = roleScopeLabel(active, names);

  return (
    <section className="analysis-section" aria-label="Dynamic analysis tasks">
      <h3 className="eyebrow">Analysis tasks</h3>
      <p className="subtext mt-2">
        {progress.screening.resolved} of{" "}
        {sourceColumnCount || progress.screening.columns}{" "}
        {sourceColumnCount ? "source columns" : "recorded columns"} screened.
      </p>
      <p className="subtext mt-2">
        {progress.resolvedRoots} of {progress.roots} original tasks resolved;{" "}
        {progress.unresolvedRoots} unresolved. {progress.resolvedObligations} of{" "}
        {progress.obligations} recorded obligations resolved.
      </p>
      <p className="subtext mt-2">
        {progress.tasks} saved tasks · {progress.attempts} attempts reserved. A
        column can have several role tasks for different assignment events, and
        one task can own several effect obligations. A widened task resolves
        when its replacement is accepted.
      </p>
      <label className="block mt-3" htmlFor={selectorId}>
        Inspect a saved task
      </label>
      <select
        id={selectorId}
        name="analysis-task"
        className="context-input mt-2 w-full min-w-0"
        value={active.id}
        onChange={(event) => setSelected(event.target.value)}
      >
        {tasks.map((task, index) => {
          const label = [
            task.kind.replaceAll("_", " "),
            task.checkpoint,
            task.column_ids.map((id) => names[id] ?? id).join(", "),
            roleScopeLabel(task, names),
            task.status,
          ]
            .filter(Boolean)
            .join(" / ");
          return (
            <option key={task.id} value={task.id}>
              {index + 1}. {label}
            </option>
          );
        })}
      </select>
      <article
        key={active.id}
        className="checkpoint-inspector min-w-0 break-words"
        aria-label={`${active.kind.replaceAll("_", " ")} task inspector`}
      >
        <p className={`gate-status state-${active.status}`} role="status">
          {active.status.toUpperCase()} / {active.checkpoint}
        </p>
        <h4 className="eyebrow mt-3">Why this task exists</h4>
        <p className="mt-2 whitespace-pre-wrap">{active.reason}</p>
        <p className="subtext mt-3 tabular-nums">
          {active.attempt_details.length} of {active.allowance.attempts_limit}{" "}
          attempts reserved; {active.attempts.length} gate receipts saved
        </p>
        <details className="mt-3" open>
          <summary>Scope and widening</summary>
          {activeRoleScope && (
            <p className="subtext mt-2 [overflow-wrap:anywhere]">
              {activeRoleScope}
            </p>
          )}
          <RecordDetails
            value={{
              task_key: active.key,
              obligation_keys: active.obligation_keys,
              scope: active.scope,
            }}
          />
          <p className="eyebrow mt-3">Columns in this task</p>
          <EvidenceLinks
            references={active.column_ids.map((id) => `columns.${id}`)}
            inspect={inspect}
            labels={labels}
          />
          <p className="subtext mt-3">
            This is the task's saved scope. Other columns remain available in
            the pinned context pack.
          </p>
        </details>
        <details className="mt-3">
          <summary>Evidence and allowance</summary>
          <EvidenceLinks references={offered} inspect={inspect} />
          <RecordDetails
            value={{
              budget_class: active.budget_kind,
              allowance: {
                attempts_per_task: active.allowance.attempts_limit,
                calls_per_column: active.allowance.column_limit,
                shared_calls: active.allowance.shared_limit,
              },
              evidence_scope: active.packet.evidence_scope,
            }}
          />
          <p className="subtext mt-3">
            Each reserved attempt charges every listed obligation. The task,
            column and shared allowances are separate limits.
          </p>
          <p className="analysis-hash" translate="no">
            Task: {active.id}
            <br />
            Input hash: {active.input_hash}
          </p>
        </details>
        <details className="mt-3">
          <summary>
            Prerequisite receipts ({active.dependency_receipt_ids.length})
          </summary>
          {active.dependency_receipt_ids.length ? (
            active.dependency_receipt_ids.map((id) => {
              const receipt = receipts.get(id);
              return receipt ? (
                <details className="mt-3" key={id}>
                  <summary>
                    {receipt.checkpoint} / {receipt.status}
                  </summary>
                  <CheckpointAttempt
                    receipt={receipt}
                    inspect={inspect}
                    run={data.id}
                  />
                </details>
              ) : (
                <p key={id} className="analysis-hash" translate="no">
                  {id}: receipt is not included in this view.
                </p>
              );
            })
          ) : (
            <p className="subtext mt-2">No prerequisite receipts recorded.</p>
          )}
        </details>
        {!active.attempts.length && (
          <p className="subtext mt-3">
            {active.status === "running"
              ? "Task in progress. No gate decision has been committed yet."
              : "No scientific gate receipt has been saved."}
          </p>
        )}
        {providerOnly.length > 0 && (
          <details className="mt-3">
            <summary>
              Provider attempts without a gate receipt ({providerOnly.length})
            </summary>
            <p className="subtext mt-2">
              These reservations are persisted and charged. They do not
              establish an accepted scientific decision.
            </p>
            <RecordDetails value={providerOnly} />
          </details>
        )}
        {active.attempts.map((attempt, index) => (
          <details key={attempt.id} open={index === active.attempts.length - 1}>
            <summary>
              Attempt {index + 1} · {attempt.status}
            </summary>
            <CheckpointAttempt
              receipt={attempt}
              inspect={inspect}
              run={data.id}
              telemetry={
                active.attempt_details.find(
                  (detail) => detail.receipt_id === attempt.id,
                )?.metrics
              }
            />
          </details>
        ))}
      </article>
    </section>
  );
}
