import type { ReactNode } from "react";
import { AlertCircle, LoaderCircle } from "lucide-react";
import { errorMessage } from "@/lib/api/client";

export function ErrorMessage({ error }: { error: unknown }) {
  return (
    <div className="error-box" role="alert">
      <AlertCircle size={14} className="inline mr-2" aria-hidden="true" />
      {errorMessage(error)}
    </div>
  );
}

export function Loading({
  children = "Loading workspace...",
}: {
  children?: ReactNode;
}) {
  return (
    <div
      role="status"
      className="flex items-center gap-3 p-8 text-muted-foreground"
    >
      <LoaderCircle size={16} className="animate-spin" aria-hidden="true" />
      {children}
    </div>
  );
}
