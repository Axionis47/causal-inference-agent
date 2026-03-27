import { useRef, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation } from "@tanstack/react-query";
import { useBlocker } from "@tanstack/react-router";
import { Send } from "lucide-react";
import { client, errorMessage } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { ErrorMessage } from "@/components/feedback";

const schema = z.object({
  text: z
    .string()
    .trim()
    .min(1, "Write a message before sending.")
    .max(8000, "Use at most 8,000 characters."),
});

export function Composer({
  conversationId,
  latest,
  busy,
  selected,
  onSent,
}: {
  conversationId: string;
  latest: number;
  busy: boolean;
  selected: string[];
  onSent: () => Promise<void>;
}) {
  const [mode, setMode] = useState<"chat" | "context">("chat");
  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues: { text: "" },
  });
  const submission = useRef<{
    signature: string;
    id: string;
    expectedTurn: number;
  } | null>(null);
  const dirty = form.formState.isDirty;
  useBlocker({
    shouldBlockFn: ({ current, next }) =>
      dirty &&
      (next.pathname !== current.pathname ||
        !("conversation" in next.search) ||
        next.search.conversation !== conversationId) &&
      !window.confirm("Discard your unsent message?"),
    enableBeforeUnload: dirty,
  });
  const send = useMutation({
    mutationFn: async ({ text }: z.infer<typeof schema>) => {
      const signature = JSON.stringify([text, selected, mode]);
      if (submission.current?.signature !== signature)
        submission.current = {
          signature,
          id: crypto.randomUUID(),
          expectedTurn: latest,
        };
      const result = await client.POST(
        "/api/conversations/{conversation_id}/turns",
        {
          params: { path: { conversation_id: conversationId } },
          body: {
            text,
            expected_turn: submission.current.expectedTurn,
            artifact_ids: selected,
            submission_id: submission.current.id,
            mode,
          },
        },
      );
      if (result.error) {
        // A rejected stale request can be resubmitted after a refresh. Ambiguous
        // network failures keep their original identity and expected turn.
        if (result.response.status === 409) submission.current = null;
        throw new Error(errorMessage(result.error));
      }
    },
    onSuccess: async () => {
      form.reset();
      submission.current = null;
      await onSent();
    },
  });
  return (
    <form
      className="chat-composer"
      onSubmit={form.handleSubmit((values) => {
        if (!busy && !send.isPending) send.mutate(values);
      })}
    >
      <div className="flex gap-2 mb-3" role="group" aria-label="Message mode">
        <Button
          size="sm"
          variant={mode === "chat" ? "secondary" : "ghost"}
          type="button"
          aria-pressed={mode === "chat"}
          onClick={() => setMode("chat")}
          disabled={send.isPending}
        >
          Chat
        </Button>
        <Button
          size="sm"
          variant={mode === "context" ? "secondary" : "ghost"}
          type="button"
          aria-pressed={mode === "context"}
          onClick={() => setMode("context")}
          disabled={send.isPending}
        >
          Update context
        </Button>
      </div>
      <label htmlFor="chat-message" className="eyebrow">
        Your message
      </label>
      <textarea
        id="chat-message"
        className="context-textarea mt-3"
        rows={3}
        placeholder="Ask about the data, request a visualization, or follow up…"
        autoComplete="off"
        readOnly={send.isPending}
        aria-invalid={!!form.formState.errors.text}
        aria-describedby={
          form.formState.errors.text ? "message-error" : "message-hint"
        }
        {...form.register("text")}
        onKeyDown={(event) => {
          if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.nativeEvent.isComposing &&
            event.keyCode !== 229
          ) {
            event.preventDefault();
            if (!event.repeat && !busy && !send.isPending)
              event.currentTarget.form?.requestSubmit();
          }
        }}
      />
      {form.formState.errors.text && (
        <p id="message-error" className="field-error">
          {form.formState.errors.text.message}
        </p>
      )}
      {send.error && <ErrorMessage error={send.error} />}
      <div className="flex items-center justify-between flex-wrap gap-3 mt-3">
        <p id="message-hint" className="subtext">
          {selected.length ? `${selected.length} artifact(s) selected. ` : ""}
          {busy
            ? "A reply is in progress. You can draft your next message."
            : mode === "context"
              ? "Debate or propose changes. Apply changes saves them."
              : "Saved history and selected context carry into each turn."}{" "}
          Enter to send · Shift+Enter for a new line.
        </p>
        <Button type="submit" disabled={busy || send.isPending}>
          <Send size={14} aria-hidden="true" />
          {send.isPending ? "Sending…" : "Send message"}
        </Button>
      </div>
    </form>
  );
}
