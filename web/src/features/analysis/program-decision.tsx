import { useEffect, useId, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";

const rejection = z.object({
  reason: z
    .string()
    .trim()
    .min(1, "Give a reason for rejecting this program.")
    .max(4000),
});

export type ProgramDecisionInput =
  { kind: "approve" } | { kind: "reject" | "revise"; reason: string };

export function ProgramDecision({
  revision,
  pending,
  decide,
}: {
  revision: number;
  pending: "approve" | "reject" | "revise" | null;
  decide: (decision: ProgramDecisionInput) => void;
}) {
  const [reviewAction, setReviewAction] = useState<"reject" | "revise" | null>(
    null,
  );
  const rejecting = reviewAction !== null;
  const reasonId = useId();
  const form = useForm<z.infer<typeof rejection>>({
    resolver: zodResolver(rejection),
    defaultValues: { reason: "" },
  });
  useEffect(() => {
    if (rejecting) form.setFocus("reason");
  }, [rejecting, form]);
  return (
    <section
      className="analysis-approval"
      aria-label="Program approval decision"
    >
      <h3 className="eyebrow">Approval required</h3>
      <p className="mt-2">
        {revision
          ? "This revised program preserves the scientific snapshot. Review the diff before authorizing execution."
          : "Review the target, treatments, graph, assumptions, specification and code before authorizing execution."}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button
          type="button"
          disabled={pending !== null || rejecting}
          onClick={() => decide({ kind: "approve" })}
        >
          {pending === "approve"
            ? "Recording approval…"
            : revision
              ? "Approve revised program"
              : "Approve frozen analysis"}
        </Button>
        {!rejecting && revision < 2 && (
          <Button
            type="button"
            variant="outline"
            disabled={pending !== null}
            onClick={() => setReviewAction("revise")}
          >
            Request correction
          </Button>
        )}
        {!rejecting && (
          <Button
            type="button"
            variant="outline"
            disabled={pending !== null}
            onClick={() => setReviewAction("reject")}
          >
            Reject this program
          </Button>
        )}
      </div>
      {rejecting && (
        <form
          className="mt-4 space-y-3"
          onSubmit={form.handleSubmit(({ reason }) =>
            decide({ kind: reviewAction!, reason }),
          )}
        >
          <p className="subtext">
            {reviewAction === "revise"
              ? "Send this feedback to the computation agent. It will revise the program within its remaining allowance and return for fresh approval."
              : "Rejection saves your reason and stops this run. The frozen science and program remain available for inspection."}
          </p>
          <label className="block" htmlFor={reasonId}>
            {reviewAction === "revise"
              ? "Correction needed"
              : "Reason for rejection"}
          </label>
          <textarea
            id={reasonId}
            className="context-textarea w-full"
            rows={3}
            maxLength={4000}
            placeholder="Describe the issue found during review…"
            aria-invalid={!!form.formState.errors.reason}
            aria-describedby={`${reasonId}-error`}
            disabled={pending !== null}
            {...form.register("reason")}
          />
          <p id={`${reasonId}-error`} className="field-error" role="status">
            {form.formState.errors.reason?.message}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button type="submit" variant="outline" disabled={pending !== null}>
              {pending !== null
                ? "Saving review…"
                : reviewAction === "revise"
                  ? "Send correction request"
                  : "Save rejection and stop"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              disabled={pending !== null}
              onClick={() => {
                form.reset();
                setReviewAction(null);
              }}
            >
              Cancel review
            </Button>
          </div>
        </form>
      )}
    </section>
  );
}
