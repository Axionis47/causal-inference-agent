import { useId, useState } from "react";
import type { components } from "@/lib/api/schema";
import { EvidenceLinks, RecordDetails } from "./inspector";

type Admission = components["schemas"]["ClaimAdmission"];
type Assumption = components["schemas"]["Assumption"];
type Analysis = components["schemas"]["AnalysisView"];

export function AuthorityInspector({
  data,
  inspect,
}: {
  data: Analysis;
  inspect: (reference: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const id = useId();
  const authority = data.authority;
  if (!authority)
    return (
      <p className="subtext mt-3" aria-label="Claim authority availability">
        Claim admissions and assumption lifecycle are unavailable in this
        historical record. Existing decisions and provider records remain
        inspectable.
      </p>
    );
  const claims = authority.claims ?? [];
  const ledger = authority.ledger ?? [];
  const identity = (claim: (typeof claims)[number]) =>
    `${claim.receipt_id}:${claim.path}`;
  const chosen =
    claims.find((claim) => identity(claim) === selected) ?? claims[0];
  return (
    <section
      className="analysis-section"
      aria-label="Claim authority and assumptions"
    >
      <h3 className="eyebrow">Claim authority and assumptions</h3>
      <p className="subtext mt-2">
        Admissions record each claim’s permitted authority. Receipt status
        records whether its complete scientific decision passed. Neither proves
        that a source entails every proposed conclusion.
      </p>
      <p className="subtext mt-2">
        {claims.filter((claim) => claim.admission.status === "admitted").length}{" "}
        admitted ·{" "}
        {
          claims.filter((claim) => claim.admission.status === "candidate")
            .length
        }{" "}
        candidates ·{" "}
        {claims.filter((claim) => claim.admission.status === "rejected").length}{" "}
        rejected claim records
      </p>
      {chosen ? (
        <>
          <label className="block mt-3" htmlFor={id}>
            Inspect a claim admission
          </label>
          <select
            id={id}
            className="context-select w-full mt-2"
            value={identity(chosen)}
            onChange={(event) => setSelected(event.target.value)}
          >
            {claims.map((claim) => (
              <option key={identity(claim)} value={identity(claim)}>
                {claim.checkpoint} / {claim.admission.status} / {claim.path}
              </option>
            ))}
          </select>
          <article
            className="mt-3"
            key={identity(chosen)}
            aria-label="Claim admission inspector"
          >
            <p className="subtext">
              Decision receipt: {chosen.receipt_status.replaceAll("_", " ")}
            </p>
            <p className="analysis-hash">
              {chosen.receipt_id}
              <br />
              {chosen.path}
            </p>
            {chosen.statement && <p className="mt-2">{chosen.statement}</p>}
            <a
              className="context-link mt-2 inline-block"
              href={`/api/analyses/${data.id}/files/checkpoint-${chosen.receipt_id}.json`}
              target="_blank"
              rel="noreferrer"
            >
              Open saved decision receipt
            </a>
            <AdmissionDetails admission={chosen.admission} inspect={inspect} />
          </article>
        </>
      ) : (
        <p className="subtext mt-3">No claim admission is recorded yet.</p>
      )}
      <h4 className="eyebrow mt-5">
        Assumption ledger / {ledger.length} records
      </h4>
      <p className="subtext mt-2">
        Candidates are not premises of the analysis. Relied-upon entries are
        explicit premises. Each record retains its status at that receipt;
        earlier candidates and rejected proposals remain visible.
      </p>
      {(["relied_upon", "candidate", "rejected", null] as const).map(
        (status) => {
          const entries = ledger.filter(
            (record) => (record.entry.status ?? null) === status,
          );
          if (!entries.length) return null;
          return (
            <details className="mt-3" key={status ?? "unavailable"}>
              <summary>
                {status?.replaceAll("_", " ") ?? "Lifecycle unavailable"} /{" "}
                {entries.length}
              </summary>
              {entries.map((record) => (
                <article
                  className="checkpoint-inspector mt-3"
                  key={`${record.receipt_id}:${record.entry.id}`}
                >
                  <p className="analysis-hash">{record.entry.id}</p>
                  <p className="subtext mt-2">
                    {record.checkpoint} receipt:{" "}
                    {record.receipt_status.replaceAll("_", " ")}
                  </p>
                  <a
                    className="context-link mt-2 inline-block"
                    href={`/api/analyses/${data.id}/files/checkpoint-${record.receipt_id}.json`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open saved decision receipt
                  </a>
                  <AssumptionDetails
                    assumption={record.entry}
                    inspect={inspect}
                  />
                </article>
              ))}
            </details>
          );
        },
      )}
      {!ledger.length && (
        <p className="subtext mt-2">No assumption record yet.</p>
      )}
    </section>
  );
}

export function AdmissionDetails({
  admission,
  inspect,
}: {
  admission: Admission;
  inspect: (reference: string) => void;
}) {
  return (
    <div className="checkpoint-inspector min-w-0">
      <p className="eyebrow">
        {admission.status} / {admission.authority}
      </p>
      <p className="mt-2">{admission.reason}</p>
      <p className="subtext mt-2">
        Rule: {admission.rule} · Predicate: {admission.predicate}
      </p>
      <p className="eyebrow mt-3">Direct sources</p>
      <EvidenceLinks
        references={admission.source_refs ?? []}
        inspect={inspect}
      />
      {(admission.sources ?? []).map((source, index) => (
        <div className="mt-3 min-w-0" key={`${source.reference}:${index}`}>
          <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">
            {source.statement}
          </p>
          <p className="subtext mt-2">
            {source.basis.replaceAll("_", " ")} / {source.status}
          </p>
          {source.qualifications && (
            <p className="subtext mt-2">
              Qualifications: {source.qualifications}
            </p>
          )}
          {source.conditions && (
            <p className="subtext mt-2">Conditions: {source.conditions}</p>
          )}
          {source.timing && (
            <p className="subtext mt-2">Timing: {source.timing}</p>
          )}
        </div>
      ))}
      {admission.ledger_id && (
        <div className="mt-3">
          <p className="eyebrow">Assumption record</p>
          <EvidenceLinks
            references={[`assumption:${admission.ledger_id}`]}
            inspect={inspect}
          />
        </div>
      )}
      <details className="mt-3">
        <summary>Exact claim scope</summary>
        <RecordDetails value={admission.scope} />
      </details>
      <details className="mt-3">
        <summary>Original proposal, before admission</summary>
        <p className="subtext mt-2">
          This is the saved typed proposal. Exact provider text, when available,
          is recorded separately with the provider attempt.
        </p>
        <RecordDetails value={admission.original_proposal} />
      </details>
    </div>
  );
}

export function AssumptionDetails({
  assumption,
  inspect,
}: {
  assumption: Assumption;
  inspect: (reference: string) => void;
}) {
  return (
    <div className="min-w-0">
      <p className="eyebrow">
        {assumption.status?.replaceAll("_", " ") ??
          "Lifecycle status unavailable in this record"}
      </p>
      <p className="mt-2">{assumption.statement}</p>
      <p className="subtext mt-2">{assumption.rationale}</p>
      {assumption.admission_reason &&
        assumption.admission_reason !== assumption.rationale && (
          <p className="mt-2">{assumption.admission_reason}</p>
        )}
      <p className="subtext mt-2">Affects: {assumption.affects.join(", ")}</p>
      {assumption.required_obligation && (
        <p className="subtext mt-2">
          Required obligation: {assumption.required_obligation}
        </p>
      )}
      <EvidenceLinks
        references={assumption.depends_on ?? []}
        inspect={inspect}
      />
      {assumption.scope && (
        <details className="mt-3">
          <summary>Exact assumption scope</summary>
          <RecordDetails value={assumption.scope} />
        </details>
      )}
      {assumption.original_proposal && (
        <details className="mt-3">
          <summary>Original assumption proposal</summary>
          <RecordDetails value={assumption.original_proposal} />
        </details>
      )}
    </div>
  );
}
