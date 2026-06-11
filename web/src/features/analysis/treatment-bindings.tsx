import type { ReactNode } from "react";
import type { components } from "@/lib/api/schema";
import { EvidenceLinks } from "./inspector";

type Analysis = components["schemas"]["AnalysisView"];
type Treatments = components["schemas"]["Treatments"];
type Claim = components["schemas"]["Claim"];
type TypedValue = components["schemas"]["TypedValue"];
type Binding = components["schemas"]["ContrastBinding"];
type Value = components["schemas"]["TreatmentValue"];
type Inspect = (key: string) => void;

function TypedLiteral({ value }: { value: TypedValue }) {
  return (
    <span>
      {value.kind}:{" "}
      <code className="whitespace-pre-wrap" translate="no">
        {value.kind === "text" ? JSON.stringify(value.literal) : value.literal}
      </code>
    </span>
  );
}

function BindingEvidence({
  claim,
  inspect,
}: {
  claim: Claim;
  inspect: Inspect;
}) {
  return (
    <div>
      <p>{claim.statement}</p>
      <p className="subtext mt-1">Basis: {claim.basis}</p>
      <EvidenceLinks references={claim.references ?? []} inspect={inspect} />
      {!!claim.prerequisites?.length && (
        <div className="mt-2">
          <p className="subtext">Decision prerequisites</p>
          <EvidenceLinks references={claim.prerequisites} inspect={inspect} />
        </div>
      )}
    </div>
  );
}

function BindingTable({
  caption,
  headings,
  children,
}: {
  caption: string;
  headings: string[];
  children: ReactNode;
}) {
  return (
    <table className="mt-3 w-full table-fixed border-collapse text-xs [overflow-wrap:anywhere] [&_th]:border-b [&_th]:border-[var(--border)] [&_th]:py-2 [&_th]:pr-2 [&_th]:text-left [&_th]:align-top [&_td]:border-b [&_td]:border-[var(--border)] [&_td]:py-3 [&_td]:pr-2 [&_td]:align-top">
      <caption className="subtext mb-2 text-left">{caption}</caption>
      <thead>
        <tr>
          {headings.map((heading) => (
            <th scope="col" key={heading}>
              {heading}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  );
}

function BoundValue({
  id,
  values,
  names,
}: {
  id: string;
  values: ReadonlyMap<string, Value>;
  names: ReadonlyMap<string, string>;
}) {
  const value = values.get(id);
  return value ? (
    <div>
      <span>{names.get(value.variable_id) ?? value.variable_id}: </span>
      <TypedLiteral value={value.value} />
      <p className="analysis-hash" translate="no">
        {id}
      </p>
    </div>
  ) : (
    <span>
      Value unavailable: <code translate="no">{id}</code>
    </span>
  );
}

function Comparison({
  binding,
  values,
  names,
  inspect,
}: {
  binding: Binding;
  values: ReadonlyMap<string, Value>;
  names: ReadonlyMap<string, string>;
  inspect: Inspect;
}) {
  const bound = (id: string | undefined) =>
    id ? (
      <BoundValue key={id} id={id} values={values} names={names} />
    ) : (
      <span>Value unavailable in this saved record.</span>
    );
  let rows: [string, ReactNode][] = [];
  if (binding.kind === "group_time" && binding.group_time) {
    const group = binding.group_time;
    rows = [
      ["Treated cohort", bound(group.cohort_value_id)],
      ["Comparison cohorts", group.comparison_value_ids.map(bound)],
      ["Time variable", <code translate="no">{group.time_variable_id}</code>],
      [
        "Adoption period",
        group.adoption_period ? (
          <TypedLiteral value={group.adoption_period} />
        ) : (
          "Unavailable in this saved record."
        ),
      ],
      ["Evaluation period", <TypedLiteral value={group.period} />],
      ["Baseline period", <TypedLiteral value={group.baseline} />],
    ];
  } else if (binding.kind === "discontinuity" && binding.discontinuity) {
    const jump = binding.discontinuity;
    rows = [
      [
        "Running variable",
        <code translate="no">{jump.running_variable_id}</code>,
      ],
      ["Cutoff", <TypedLiteral value={jump.cutoff} />],
      [
        "Equality belongs to",
        jump.cutoff_in === "above" ? "Above (>= cutoff)" : "Below (<= cutoff)",
      ],
      ["Direction", jump.direction.replaceAll("_", " ")],
      ["Estimand", jump.estimand.replaceAll("_", " ")],
      ["Control level", bound(jump.treatment_value_ids[0])],
      ["Treated level", bound(jump.treatment_value_ids[1])],
      ...(jump.assignment_value_ids?.length
        ? ([
            ["Ineligible level", bound(jump.assignment_value_ids[0])],
            ["Eligible level", bound(jump.assignment_value_ids[1])],
          ] as [string, ReactNode][])
        : [
            [
              "Eligibility",
              "Explicit threshold rule; no observed eligibility levels recorded",
            ] as [string, ReactNode],
          ]),
    ];
  }
  return (
    <>
      <p className="subtext mt-3">
        Comparison kind: {binding.kind.replaceAll("_", " ")}
      </p>
      <BindingEvidence claim={binding.evidence} inspect={inspect} />
      {binding.kind === "static" && binding.static ? (
        <BindingTable
          caption="Ordered signed contrast over complete factor states"
          headings={["Term / weight", "Exact state"]}
        >
          {binding.static.terms.map((term, index) => (
            <tr key={index}>
              <th scope="row" className="font-normal tabular-nums">
                {index + 1} / {term.weight > 0 ? "+" : ""}
                {term.weight}
              </th>
              <td>{term.value_ids.map(bound)}</td>
            </tr>
          ))}
        </BindingTable>
      ) : binding.kind === "aggregate" && binding.aggregate ? (
        <BindingTable
          caption="Declared component weights"
          headings={["Effect", "Weight"]}
        >
          {binding.aggregate.terms.map((term) => (
            <tr key={term.effect_id}>
              <th scope="row" className="font-normal" translate="no">
                {term.effect_id}
              </th>
              <td className="tabular-nums">{term.weight}</td>
            </tr>
          ))}
        </BindingTable>
      ) : binding.kind === "unsupported" && binding.unsupported ? (
        <div className="mt-3">
          <BindingEvidence claim={binding.unsupported} inspect={inspect} />
        </div>
      ) : rows.length ? (
        <BindingTable
          caption="Exact comparison selectors"
          headings={["Selector", "Saved value"]}
        >
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th scope="row" className="font-normal">
                {label}
              </th>
              <td>{value}</td>
            </tr>
          ))}
        </BindingTable>
      ) : (
        <p className="subtext mt-3">
          Typed comparison payload unavailable in this saved record.
        </p>
      )}
    </>
  );
}

export function TreatmentBindings({
  data,
  inspect,
}: {
  data: Analysis;
  inspect: Inspect;
}) {
  const frame = data.snapshot?.frame ?? data.frame;
  const receipt = data.checkpoints
    .find((checkpoint) => checkpoint.id === "treatments")
    ?.attempts.filter(
      (attempt) => attempt.status === "accepted" && attempt.complete !== false,
    )
    .at(-1);
  // Accepted treatment receipts contain the validated Treatments contract.
  const treatments =
    frame?.treatments ?? (receipt?.output as Treatments | undefined);
  const values = treatments?.values ?? [];
  const effects = treatments?.effects ?? [];
  const factors = treatments?.treatments ?? [];
  const names = new Map([
    ...factors.map(({ variable }) => [variable.id, variable.label] as const),
    ...(frame?.roles.assignments ?? []).map(
      ({ variable }) => [variable.id, variable.label] as const,
    ),
  ]);
  const registry = new Map(values.map((value) => [value.id, value]));
  return (
    <details
      open
      className="analysis-section min-w-0"
      aria-label="Treatment values and comparisons"
    >
      <summary>Treatment values and comparisons</summary>
      <p className="subtext mt-2">
        {data.snapshot
          ? "From the frozen scientific snapshot."
          : frame
            ? "From the accepted causal frame."
            : receipt
              ? "From the accepted treatment checkpoint."
              : "No accepted treatment decision is available yet."}
      </p>
      {!!factors.length && (
        <p className="mt-3 text-xs [overflow-wrap:anywhere]">
          Factors:{" "}
          {factors
            .map(({ variable }) => `${variable.label} (${variable.id})`)
            .join(", ")}
        </p>
      )}
      {values.length ? (
        <details open className="mt-3">
          <summary>Typed value registry / {values.length}</summary>
          <BindingTable
            caption="Exact values and their recorded authority"
            headings={[
              "Variable / value ID",
              "Typed literal",
              "Meaning / evidence",
            ]}
          >
            {values.map((value) => (
              <tr key={value.id}>
                <th scope="row" className="font-normal">
                  <p>{names.get(value.variable_id) ?? "Variable"}</p>
                  <code translate="no">{value.variable_id}</code>
                  <p className="analysis-hash" translate="no">
                    {value.id}
                  </p>
                </th>
                <td>
                  <TypedLiteral value={value.value} />
                </td>
                <td>
                  <BindingEvidence claim={value.meaning} inspect={inspect} />
                </td>
              </tr>
            ))}
          </BindingTable>
        </details>
      ) : (
        <p className="subtext mt-3">
          Typed value registry unavailable in this saved record. Values are not
          inferred from prose.
        </p>
      )}
      {effects.map((effect) => (
        <details
          open
          className="mt-4 min-w-0 [overflow-wrap:anywhere]"
          key={effect.id}
        >
          <summary>
            <span translate="no">{effect.id}</span> / {effect.priority}
          </summary>
          <p className="subtext mt-2">
            Factors: {effect.treatment_ids.join(", ")}
          </p>
          <p className="mt-2">{effect.estimand}</p>
          <BindingEvidence claim={effect.contrast} inspect={inspect} />
          {effect.comparison ? (
            <Comparison
              binding={effect.comparison}
              values={registry}
              names={names}
              inspect={inspect}
            />
          ) : (
            <p className="subtext mt-3">
              Typed comparison unavailable in this saved record. No selectors or
              weights are inferred from prose.
            </p>
          )}
        </details>
      ))}
      {!effects.length && (
        <p className="subtext mt-3">
          No accepted effect comparisons are recorded.
        </p>
      )}
    </details>
  );
}
