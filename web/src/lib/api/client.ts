import createFetchClient from "openapi-fetch";
import createClient from "openapi-react-query";
import { QueryClient } from "@tanstack/react-query";
import type { paths, components } from "./schema";

export const client = createFetchClient<paths>({
  baseUrl: "",
  credentials: "same-origin",
  headers: { "X-Causal-Client": "web" },
});
client.use({
  onResponse({ request, response }) {
    if (!response.ok && !response.headers.get("content-type")?.includes("application/json")) {
      throw new Error("The server is temporarily unavailable. Please retry.");
    }
    if (
      response.status === 401 &&
      !request.url.endsWith("/api/auth/login") &&
      window.location.pathname !== "/login"
    ) {
      window.location.assign("/login");
    }
  },
});
export const api = createClient(client);
export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: false } },
});
export type Dataset = components["schemas"]["DatasetView"];

export function errorMessage(error: unknown): string {
  if (
    error &&
    typeof error === "object" &&
    "detail" in error &&
    typeof error.detail === "string"
  )
    return error.detail;
  if (error instanceof Error) return error.message;
  return "The request could not be completed. Please try again.";
}

export function fileSize(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
