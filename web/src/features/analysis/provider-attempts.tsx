import type { components } from "@/lib/api/schema";
import { RecordDetails } from "./inspector";

type Analysis = components["schemas"]["AnalysisView"];

function providerStatus(status: string) {
  if (status === "accepted") return "Proposal received";
  if (status === "uncertain") return "Response unknown";
  if (status === "failed") return "Provider failed";
  if (status === "reserved") return "Reserved";
  return status.replaceAll("_", " ");
}

export function SharedProviderAttempts({ data }: { data: Analysis }) {
  const attempts = data.shared_attempts ?? [];
  if (!attempts.length) return null;

  return (
    <section className="analysis-section" aria-label="Shared provider attempts">
      <h3 className="eyebrow">Shared provider attempts</h3>
      <p className="subtext mt-2">
        {attempts.length} shared attempts reserved, separate from column tasks.
        Every reservation is persisted and charged, including failed or unknown
        responses. A received proposal does not mean a scientific gate accepted
        it. Checkpoint decisions are recorded above.
      </p>
      {attempts.map((attempt) => {
        const responseHash = attempt.metrics.response_text_hash;
        const traceUrl =
          typeof attempt.metrics.trace_url === "string"
            ? attempt.metrics.trace_url
            : null;
        const exactText =
          attempt.has_raw_output &&
          typeof responseHash === "string" &&
          /^[a-f0-9]{64}$/i.test(responseHash);
        return (
          <details className="mt-3 min-w-0" key={attempt.id}>
            <summary className="[overflow-wrap:anywhere]">
              {attempt.number}. {attempt.logical_key} /{" "}
              {providerStatus(attempt.status)}
            </summary>
            <div className="checkpoint-inspector min-w-0">
              <p className="subtext mt-2">
                {attempt.has_proposal
                  ? "A provider proposal is saved for inspection. Its scientific decision is recorded separately."
                  : "No provider proposal is saved for this reservation."}
              </p>
              <a
                className="context-link mt-3 inline-block"
                href={`/api/analyses/${data.id}/provider-attempts/${attempt.id}`}
                target="_blank"
                rel="noreferrer"
              >
                Open saved provider record
              </a>
              <p className="subtext mt-3">
                {traceUrl ? (
                  <a
                    className="context-link"
                    href={traceUrl}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open provider attempt trace
                  </a>
                ) : (
                  "Provider attempt trace unavailable for this reservation."
                )}
              </p>
              <p className="subtext mt-2">
                {exactText
                  ? "Exact provider response text is saved with its recorded hash. The typed proposal is a separate representation."
                  : attempt.has_raw_output
                    ? "Saved output may be a typed or legacy representation. Exact provider text is unconfirmed for this record."
                    : "No response text is saved for this reservation. Inspect the saved status and any failure details."}
              </p>
              <RecordDetails
                value={{
                  reservation_id: attempt.id,
                  stored_key: attempt.key,
                  obligation_key: attempt.obligation_key,
                  provider_status: attempt.status,
                  started_at: attempt.started_at,
                  completed_at: attempt.completed_at,
                  input_hash: attempt.input_hash,
                  request_hash: attempt.request_hash,
                  saved_content: {
                    request: attempt.has_request,
                    response_representation: exactText
                      ? "Exact provider text recorded"
                      : attempt.has_raw_output
                        ? "Typed or legacy output; exact text unconfirmed"
                        : "Unavailable",
                    proposal: attempt.has_proposal,
                  },
                  metrics: {
                    ...attempt.metrics,
                    response_text_hash: responseHash ?? "Unavailable",
                    provider_model_version:
                      attempt.metrics.provider_model_version ?? "Unavailable",
                    finish_reason:
                      attempt.metrics.finish_reason ?? "Unavailable",
                    usage: attempt.metrics.usage ?? "Unavailable",
                    thinking_config:
                      attempt.metrics.thinking_config ??
                      (attempt.metrics.thinking_limit_kind ===
                      "provider_default"
                        ? "Provider default"
                        : "Unavailable"),
                  },
                }}
              />
            </div>
          </details>
        );
      })}
    </section>
  );
}
