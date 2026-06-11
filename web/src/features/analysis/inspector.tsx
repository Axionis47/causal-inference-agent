import { useId, useState } from "react";
import type { components } from "@/lib/api/schema";
import { Button } from "@/components/ui/button";
import { taskProgress } from "./task-progress";

type Analysis = components["schemas"]["AnalysisView"];
type Checkpoint = components["schemas"]["CheckpointView"];
type Receipt = components["schemas"]["CheckpointReceipt"];
const explanationLabels: Readonly<Record<string, string>> = {
  summary: "Proposed explanation",
  changes: "Proposed change explanation",
  permits: "Proposed next-step explanation",
  identification: "Proposed identification explanation",
};

export function SavedAnalysisVersion({ data }: { data: Analysis }) {
  return (
    <p className="subtext mt-2" aria-label="Saved analysis versions">
      Analysis contract:{" "}
      <span translate="no">
        {data.contract ?? "Unavailable in the saved record"}
      </span>
      {data.snapshot?.input_projection && (
        <>
          <br />
          Execution input policy:{" "}
          <span translate="no">{data.snapshot.input_projection.policy}</span>
        </>
      )}
    </p>
  );
}

export function RecordDetails({ value }: { value: unknown }) {
  if (value === null || value === undefined)
    return <span className="subtext">Not supplied</span>;
  if (typeof value !== "object")
    return (
      <span className="whitespace-pre-wrap [overflow-wrap:anywhere]">
        {String(value)}
      </span>
    );
  if (Array.isArray(value))
    return value.length ? (
      <ul className="analysis-list">
        {value.map((item, i) => (
          <li key={i}>
            <RecordDetails value={item} />
          </li>
        ))}
      </ul>
    ) : (
      <span className="subtext">None</span>
    );
  return (
    <dl className="analysis-list">
      {Object.entries(value).map(([key, item]) => (
        <div key={key}>
          <dt>{explanationLabels[key] ?? key.replaceAll("_", " ")}</dt>
          <dd>
            {key === "outcome_proxy" && item === null ? (
              <span className="subtext">
                Unknown; withheld from diagnostics
              </span>
            ) : (
              <RecordDetails value={item} />
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function entries(
  value: unknown,
  path = "",
): [string, Record<string, unknown>][] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const record = value as Record<string, unknown>;
  if (typeof record.value === "string" && typeof record.basis === "string")
    return [[path, record]];
  return Object.entries(record).flatMap(([key, child]) =>
    entries(child, path ? `${path}.${key}` : key),
  );
}

function resolve(value: unknown, key: string): unknown {
  let current = value;
  for (const part of key.split(".")) {
    if (!current || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

export function EvidenceLinks({
  references,
  inspect,
  labels,
}: {
  references: string[];
  inspect: (key: string) => void;
  labels?: Readonly<Record<string, string>>;
}) {
  return (
    <div className="evidence-links">
      {references.length ? (
        references.map((key) => (
          <button
            type="button"
            className="context-link"
            key={key}
            title={key}
            onClick={() => inspect(key)}
          >
            {labels?.[key] ?? (key.length > 65 ? `…${key.slice(-64)}` : key)}
          </button>
        ))
      ) : (
        <span className="subtext">None recorded</span>
      )}
    </div>
  );
}

export function ContextPack({
  data,
  selected,
  select,
}: {
  data: Analysis;
  selected: string | null;
  select: (key: string | null) => void;
}) {
  const [search, setSearch] = useState("");
  const id = useId();
  const records = entries(data.context.content);
  const filtered = records.filter(([key, value]) =>
    `${key} ${value.value}`.toLowerCase().includes(search.toLowerCase()),
  );
  const attempts = [
    ...new Map(
      [
        ...data.checkpoints.flatMap((c) => c.attempts),
        ...(data.tasks ?? []).flatMap((task) => task.attempts),
      ].map((receipt) => [receipt.id, receipt]),
    ).values(),
  ];
  const assumptions = [
    ...attempts.filter((r) => r.status === "accepted"),
    ...attempts.filter((r) => r.status !== "accepted").reverse(),
  ].flatMap((r) => r.assumptions);
  const decision = selected?.startsWith("decision:")
    ? data.checkpoints
        .find(
          (c) =>
            c.id ===
            (selected === "decision:plan" ? "freeze" : selected.slice(9)),
        )
        ?.attempts.filter((r) => r.status === "accepted")
        .at(-1)?.output
    : undefined;
  const execution =
    selected === "execution"
      ? data.programs.at(-1)?.execution
      : selected?.startsWith("execution:")
        ? data.programs.find((p) => `execution:${p.revision}` === selected)
            ?.execution
        : undefined;
  const ledgerRecords = selected?.startsWith("assumption:")
    ? data.authority?.ledger?.filter(
        (record) => `assumption:${record.entry.id}` === selected,
      )
    : undefined;
  const record =
    selected === "question"
      ? data.question
      : selected?.startsWith("assumption:")
        ? data.authority
          ? ledgerRecords?.length
            ? ledgerRecords
            : undefined
          : assumptions.find((a) => `assumption:${a.id}` === selected)
        : selected?.startsWith("diagnostic:")
          ? data.diagnostics.find((d) => d.reference === selected)
          : (execution ??
            decision ??
            (selected ? resolve(data.context.content, selected) : undefined));
  return (
    <section
      className="analysis-section"
      aria-label="Pinned context and evidence"
    >
      <h3 className="eyebrow">Pinned context pack</h3>
      <p className="analysis-hash">
        Revision:{" "}
        {data.context_revision_id ?? "Question and dataset identities only"}
        <br />
        Pack hash: {String(data.context.hash)}
      </p>
      {selected && (
        <div className="evidence-inspector" tabIndex={0}>
          <div className="flex justify-between gap-2">
            <h4 className="eyebrow">Selected evidence</h4>
            <Button variant="ghost" onClick={() => select(null)}>
              Close
            </Button>
          </div>
          <p className="analysis-hash">{selected}</p>
          {record === undefined ? (
            <p className="subtext">
              This reference has no inspectable record in this snapshot.
            </p>
          ) : (
            <RecordDetails value={record} />
          )}
        </div>
      )}
      <details open={!!selected}>
        <summary>
          Browse exact statements and provenance ({records.length})
        </summary>
        <label className="block mt-3" htmlFor={id}>
          Find a context key or statement
        </label>
        <input
          id={id}
          name="context-evidence"
          autoComplete="off"
          className="context-input w-full mt-2"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Meaning, timing, assignment…"
        />
        <ul className="context-evidence-results">
          {filtered.slice(0, 50).map(([key, value]) => (
            <li key={key}>
              <button
                className="context-link"
                type="button"
                onClick={() => select(key)}
              >
                {key}
              </button>
              <p>{String(value.value)}</p>
              <small>
                {String(value.basis)} / {String(value.status)}
              </small>
            </li>
          ))}
        </ul>
        {filtered.length > 50 && (
          <p className="subtext">
            Showing the first 50 matching statements. Refine the search to
            inspect another key.
          </p>
        )}
        {!records.length && (
          <p className="subtext mt-3">
            No context claims were supplied. The user's question remains
            available as evidence.
          </p>
        )}
        <details className="mt-3">
          <summary>Column identities</summary>
          <RecordDetails
            value={Object.values(
              (data.context.content as Record<string, unknown> | undefined)
                ?.columns ?? {},
            ).map((column) => (column as Record<string, unknown>).identity)}
          />
        </details>
      </details>
    </section>
  );
}

function WorkflowGraph({
  checkpoints,
  tasks,
}: {
  checkpoints: Checkpoint[];
  tasks: NonNullable<Analysis["tasks"]>;
}) {
  const marker = useId().replaceAll(":", "");
  const width = 640;
  const hasTasks = tasks.length > 0;
  const step = hasTasks ? 92 : 64;
  const halfNodeHeight = hasTasks ? 35 : 19;
  const height = checkpoints.length * step + 20;
  const position = new Map(
    checkpoints.map((c, i) => [
      c.id,
      { x: 180 + (i % 2) * 160, y: halfNodeHeight + 16 + i * step },
    ]),
  );
  const progress = new Map(
    checkpoints.map((checkpoint) => [
      checkpoint.id,
      taskProgress(tasks.filter((task) => task.checkpoint === checkpoint.id)),
    ]),
  );
  return (
    <details className="mt-3">
      <summary>Workflow dependency graph</summary>
      <svg
        className="workflow-graph"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Checkpoint prerequisites, gate states and dynamic task coverage"
      >
        <desc>
          {checkpoints
            .map((checkpoint) => {
              const count = progress.get(checkpoint.id)!;
              return `${checkpoint.title}: ${checkpoint.status}; ${count.resolvedRoots} of ${count.roots} original tasks resolved, ${count.resolvedObligations} of ${count.obligations} recorded obligations resolved, ${count.screening.resolved} of ${count.screening.columns} recorded columns screened, ${count.widened} widened, ${count.tasks} saved tasks, ${count.attempts} attempts.`;
            })
            .join(" ")}
        </desc>
        <defs>
          <marker
            id={marker}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>
        {checkpoints.flatMap((c) =>
          c.prerequisites.map((parent) => {
            const a = position.get(parent);
            const b = position.get(c.id);
            return a && b ? (
              <path
                key={`${parent}:${c.id}`}
                d={`M ${a.x} ${a.y + halfNodeHeight - 1} Q ${75 + checkpoints.findIndex((v) => v.id === parent) * 12} ${(a.y + b.y) / 2} ${b.x} ${b.y - halfNodeHeight - 1}`}
                fill="none"
                stroke="currentColor"
                opacity=".5"
                markerEnd={`url(#${marker})`}
              />
            ) : null;
          }),
        )}
        {checkpoints.map((c) => {
          const p = position.get(c.id)!;
          const count = progress.get(c.id)!;
          return (
            <g key={c.id} transform={`translate(${p.x},${p.y})`}>
              <rect
                x="-140"
                y={-halfNodeHeight}
                width="280"
                height={halfNodeHeight * 2}
                className={`workflow-node state-${c.status}`}
              />
              <text
                textAnchor="middle"
                y={hasTasks ? -18 : -2}
                fontSize="12"
                fill="currentColor"
              >
                {c.title}
              </text>
              <text
                textAnchor="middle"
                y={hasTasks ? -4 : 12}
                fontSize="9"
                fill="currentColor"
              >
                {c.status.toUpperCase()}
              </text>
              {hasTasks && (
                <text
                  textAnchor="middle"
                  y="11"
                  fontSize="9"
                  fill="currentColor"
                >
                  {count.tasks
                    ? count.screening.columns
                      ? `${count.screening.resolved}/${count.screening.columns} columns screened · ${count.widened} widened`
                      : `${count.resolvedRoots}/${count.roots} tasks resolved · ${count.resolvedObligations}/${count.obligations} obligations`
                    : "No dynamic tasks recorded"}
                </text>
              )}
              {count.tasks > 0 && (
                <text
                  textAnchor="middle"
                  y="25"
                  fontSize="9"
                  fill="currentColor"
                >
                  {count.tasks} saved tasks · {count.attempts} attempts
                  {!count.screening.columns && ` · ${count.widened} widened`}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </details>
  );
}

export function CheckpointAttempt({
  receipt,
  inspect,
  run,
  telemetry,
}: {
  receipt: Receipt;
  run: string;
  inspect: (key: string) => void;
  telemetry?: unknown;
}) {
  const output = receipt.output;
  const seconds =
    (Date.parse(receipt.completed_at) - Date.parse(receipt.started_at)) / 1000;
  return (
    <article className="checkpoint-attempt">
      <p className={`gate-status state-${receipt.status}`}>
        {receipt.status.toUpperCase()} /{" "}
        {receipt.complete ? "checkpoint" : "subdecision"}
      </p>
      {!!output.summary && <p className="eyebrow mt-3">Proposed explanation</p>}
      <p className="mt-2">
        {String(output.summary ?? output.error ?? "Saved decision")}
      </p>
      {!!output.changes && (
        <p className="mt-2">
          <strong>Proposed change explanation: </strong>
          {String(output.changes)}
        </p>
      )}
      {!!output.permits && (
        <p className="mt-2">
          <strong>Proposed next-step explanation: </strong>
          {String(output.permits)}
        </p>
      )}
      {!!(output.summary || output.changes || output.permits) && (
        <p className="subtext mt-2">
          Narrative explanations are preserved for inspection. Claim admissions,
          assumption status and gate results record the decision’s authority.
        </p>
      )}
      {Array.isArray(output.unresolved) && output.unresolved.length > 0 && (
        <div className="checkpoint-blocked">
          <strong>Unresolved</strong>
          <RecordDetails value={output.unresolved} />
        </div>
      )}
      <ul className="gate-checks">
        {receipt.checks.map((check) => (
          <li key={check.rule} data-passed={check.passed}>
            <strong>{check.passed ? "PASS" : "STOP"}</strong>{" "}
            {check.rule.replaceAll("_", " ")} <small>({check.kind})</small>
            <p>{check.detail}</p>
          </li>
        ))}
      </ul>
      <p className="eyebrow mt-4">Cited evidence</p>
      <EvidenceLinks references={receipt.cited} inspect={inspect} />
      {receipt.assumptions.length > 0 && (
        <>
          <p className="eyebrow mt-4">Assumption records in this receipt</p>
          <ul className="analysis-list">
            {receipt.assumptions.map((assumption) => (
              <li key={assumption.id}>
                <EvidenceLinks
                  references={[`assumption:${assumption.id}`]}
                  inspect={inspect}
                />
                <p className="subtext">
                  {assumption.status?.replaceAll("_", " ") ??
                    "Lifecycle status unavailable in this record"}
                </p>
              </li>
            ))}
          </ul>
        </>
      )}
      <details className="mt-3">
        <summary>Decision details</summary>
        <RecordDetails
          value={Object.fromEntries(
            Object.entries(output).filter(
              ([key]) =>
                ![
                  "summary",
                  "changes",
                  "permits",
                  "citations",
                  "assumptions",
                  "unresolved",
                ].includes(key),
            ),
          )}
        />
      </details>
      <details className="mt-3">
        <summary>Evidence offered and retrieved</summary>
        {receipt.evidence_scope ? (
          <div className="mt-3">
            <p className="eyebrow">Bounded evidence packet</p>
            <RecordDetails value={receipt.evidence_scope} />
            <p className="muted">
              Other columns remain in the context pack and their own checkpoint
              receipts.
            </p>
          </div>
        ) : (
          <p className="muted mt-3">
            Packet size telemetry unavailable for this receipt.
          </p>
        )}
        <p className="eyebrow mt-3">Offered to this task</p>
        <EvidenceLinks references={receipt.offered} inspect={inspect} />
        <p className="eyebrow mt-3">Retrieved from pinned context</p>
        <EvidenceLinks references={receipt.retrieved} inspect={inspect} />
      </details>
      <details className="mt-3">
        <summary>Receipt and telemetry</summary>
        <a
          className="context-link"
          href={`/api/analyses/${run}/checkpoints/${receipt.id}/input`}
          target="_blank"
          rel="noreferrer"
        >
          Open exact task input packet
        </a>
        <dl className="analysis-list">
          <div>
            <dt>Started / completed</dt>
            <dd>
              {new Date(receipt.started_at).toLocaleString()} /{" "}
              {new Date(receipt.completed_at).toLocaleString()}
            </dd>
          </div>
          <div>
            <dt>Elapsed</dt>
            <dd>{seconds.toFixed(1)} seconds</dd>
          </div>
          <div>
            <dt>Consumption at completion</dt>
            <dd>
              <RecordDetails value={receipt.budgets} />
            </dd>
          </div>
          <div>
            <dt>Model telemetry</dt>
            <dd>
              {telemetry ? (
                <RecordDetails value={telemetry} />
              ) : (
                "Unavailable in this receipt. Inspect the task trace for any provider telemetry."
              )}
            </dd>
          </div>
          <div>
            <dt>Input / output hashes</dt>
            <dd className="analysis-hash">
              {receipt.input_hash}
              <br />
              {receipt.output_hash}
            </dd>
          </div>
          <div>
            <dt>Prerequisite receipts</dt>
            <dd className="analysis-hash">
              {receipt.prerequisite_ids.join("\n") || "None"}
            </dd>
          </div>
          <div>
            <dt>LangSmith trace</dt>
            <dd>
              {receipt.trace_url ? (
                <a
                  className="context-link"
                  href={receipt.trace_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open task trace
                </a>
              ) : (
                "Trace link unavailable"
              )}
            </dd>
          </div>
        </dl>
      </details>
    </article>
  );
}

export function CheckpointInspector({
  data,
  inspect,
}: {
  data: Analysis;
  inspect: (key: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const active =
    data.checkpoints.find((c) => c.id === selected) ??
    data.checkpoints.find((c) => c.id === data.stage) ??
    data.checkpoints[0];
  if (!active) return null;
  return (
    <section className="analysis-section" aria-label="Scientific checkpoints">
      <h3 className="eyebrow">Scientific checkpoints</h3>
      <ol className="checkpoint-list">
        {data.checkpoints.map((c, i) => (
          <li key={c.id}>
            <button
              type="button"
              aria-pressed={c.id === active.id}
              className={`checkpoint-button state-${c.status}`}
              onClick={() => setSelected(c.id)}
            >
              <span>
                {String(i + 1).padStart(2, "0")} {c.title}
              </span>
              <small>{c.status.replaceAll("_", " ")}</small>
            </button>
          </li>
        ))}
      </ol>
      <WorkflowGraph checkpoints={data.checkpoints} tasks={data.tasks ?? []} />
      <section
        className="checkpoint-inspector"
        aria-label={`${active.title} inspector`}
      >
        <p className="eyebrow">
          {active.phase} / {active.status}
        </p>
        <h4 className="mt-2">{active.question}</h4>
        {!!active.blocked_by.length && active.status === "pending" && (
          <p className="subtext mt-2">
            Requires accepted: {active.blocked_by.join(", ")}.
          </p>
        )}
        {!active.attempts.length && (
          <p className="subtext mt-3">
            {active.status === "running"
              ? "Task in progress. No gate decision has been committed yet."
              : "No decision has been submitted."}
          </p>
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
            />
          </details>
        ))}
      </section>
    </section>
  );
}
