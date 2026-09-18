import type { DatasetCreate, DatasetSummary, ProfileOut, RunFiles, SessionView } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not json */
    }
    throw new ApiError(r.status, detail);
  }
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: body === undefined ? undefined : { "content-type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  listDatasets: () => call<{ datasets: DatasetSummary[] }>("/api/datasets").then((r) => r.datasets),
  profile: (file: File) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    return call<ProfileOut>("/api/profile", { method: "POST", body: fd });
  },
  createDataset: (req: DatasetCreate) => call<DatasetSummary>("/api/datasets", json("POST", req)),
  deleteDataset: (name: string) => call<void>(`/api/datasets/${encodeURIComponent(name)}`, { method: "DELETE" }),
  session: (name: string) => call<SessionView>(`/api/sessions/${encodeURIComponent(name)}`),
  send: (name: string, text: string) => call<SessionView>(`/api/sessions/${encodeURIComponent(name)}/messages`, json("POST", { text })),
  resume: (name: string) => call<SessionView>(`/api/sessions/${encodeURIComponent(name)}/resume`, json("POST")),
  restart: (name: string) => call<SessionView>(`/api/sessions/${encodeURIComponent(name)}/restart`, json("POST")),
  runFiles: (runId: string) => call<RunFiles>(`/api/runs/${encodeURIComponent(runId)}`),
  fileUrl: (runId: string, name: string, raw = false) => `/api/runs/${encodeURIComponent(runId)}/files/${encodeURIComponent(name)}${raw ? "?raw=1" : ""}`,
  fileText: async (runId: string, name: string, raw = false) => {
    const r = await fetch(api.fileUrl(runId, name, raw));
    if (!r.ok) throw new ApiError(r.status, r.statusText);
    return r.text();
  },
};
