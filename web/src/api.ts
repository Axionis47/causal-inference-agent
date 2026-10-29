import type { IngestedDataset, DatasetList, RunFiles, ProfileView, SavedColumn, ContextView, ContextDatasetView, ContextColumnView } from "./types";
import type { components } from "./generated/schema";
type Wire<K extends keyof components["schemas"]> = components["schemas"][K];

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
      detail = typeof body.detail === "string" ? body.detail : (body.detail?.message ?? JSON.stringify(body.detail ?? body));
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
  context: (id: string) => call<ContextView>(`/api/datasets/${encodeURIComponent(id)}/context`),
  prepareContext: (id: string) => call<ContextView>(`/api/datasets/${encodeURIComponent(id)}/context`, json("POST")),
  datasetContext: (id: string, packId: string) => call<ContextDatasetView>(`/api/datasets/${encodeURIComponent(id)}/contexts/${encodeURIComponent(packId)}`),
  columnContext: (id: string, packId: string, columnId: string) =>
    call<ContextColumnView>(`/api/datasets/${encodeURIComponent(id)}/contexts/${encodeURIComponent(packId)}/columns/${encodeURIComponent(columnId)}`),
  profile: (id: string) => call<ProfileView>(`/api/datasets/${encodeURIComponent(id)}/profile`),
  generateProfile: (id: string) => call<ProfileView>(`/api/datasets/${encodeURIComponent(id)}/profile`, json("POST")),
  profileColumn: (id: string, profileId: string, columnId: string) =>
    call<SavedColumn>(`/api/datasets/${encodeURIComponent(id)}/profiles/${encodeURIComponent(profileId)}/columns/${encodeURIComponent(columnId)}`),
  listDatasets: (cursor?: string) => call<DatasetList>(`/api/datasets${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ""}`),
  dataset: (id: string) => call<IngestedDataset>(`/api/datasets/${encodeURIComponent(id)}`),
  ingest: (file: File, title: string, context: string, key: string, progress: (percent: number) => void) =>
    new Promise<IngestedDataset>((resolve, reject) => {
      const body = new FormData();
      body.append("file", file, file.name);
      body.append("title", title);
      // JSON escapes newlines so multipart serialization cannot rewrite the original account.
      body.append("context", JSON.stringify(context));
      body.append("idempotency_key", key);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/datasets");
      xhr.timeout = 180000;
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) progress(Math.round((event.loaded * 100) / event.total));
      };
      xhr.onerror = () => reject(new ApiError(0, "Connection interrupted. Retry the same submission."));
      xhr.ontimeout = () => reject(new ApiError(0, "Saving timed out. Retry the same submission to check its result."));
      xhr.onload = () => {
        try {
          const result = JSON.parse(xhr.responseText);
          if (xhr.status >= 200 && xhr.status < 300) resolve(result);
          else reject(new ApiError(xhr.status, typeof result.detail === "string" ? result.detail : "Check the form fields and retry."));
        } catch {
          reject(new ApiError(xhr.status, "Unexpected server response. Retry the same submission."));
        }
      };
      xhr.send(body);
    }),
  session: (name: string) => call<Wire<"ChatOverview">>(`/api/conversations/${encodeURIComponent(name)}`),
  command: (name: string, body: Wire<"ChatCommand">) =>
    call<Wire<"ChatOverview">>(`/api/conversations/${encodeURIComponent(name)}/commands`, json("POST", body)),
  messages: (name: string, before?: string) => call<Wire<"MessagePage">>(`/api/conversations/${name}/messages${before ? `?before=${before}` : ""}`),
  resume: (name: string) => call<Wire<"ChatOverview">>(`/api/conversations/${encodeURIComponent(name)}/resume`, json("POST")),
  chatContext: (name: string) => call<Wire<"ChatContextSummary">>(`/api/conversations/${name}/context`),
  contextChanges: (name: string, offset = 0) => call<Wire<"ContextChanges">>(`/api/conversations/${name}/context/changes?offset=${offset}`),
  chatStatements: (name: string, section: string, offset = 0) =>
    call<Wire<"ChatStatements">>(
      `/api/conversations/${name}/context/statements?${["dataset", "relationship", "requirement"].includes(section) ? `section=${section}` : `column_id=${section}`}&offset=${offset}`,
    ),
  artifacts: (name: string, earlier = false, offset = 0) =>
    call<Wire<"ArtifactSummary">[]>(`/api/conversations/${name}/artifacts?earlier=${earlier}&offset=${offset}`),
  artifact: (name: string, id: string) => call<Wire<"Artifact">>(`/api/conversations/${name}/artifacts/${id}`),
  artifactFile: (name: string, id: string, file: string) => `/api/conversations/${name}/artifacts/${id}/files/${encodeURIComponent(file)}`,
  analyses: (name: string, offset = 0) => call<Wire<"AnalysisSummary">[]>(`/api/conversations/${name}/analyses?offset=${offset}`),
  analysis: (name: string, id: string) => call<import("./types").RunView>(`/api/conversations/${name}/analyses/${id}`),
  startReview: (datasetId: string, question: string) => call<Wire<"ReviewView">>(`/api/datasets/${datasetId}/reviews`, json("POST", { question })),
  review: (datasetId: string, reviewId: string) => call<Wire<"ReviewView">>(`/api/datasets/${datasetId}/reviews/${reviewId}`),
  saveReview: (datasetId: string, reviewId: string, body: Wire<"ReviewSave">) =>
    call<Wire<"ReviewView">>(`/api/datasets/${datasetId}/reviews/${reviewId}`, json("PUT", body)),
  reviewStatements: (datasetId: string, reviewId: string, section: string, offset = 0) =>
    call<Wire<"ReviewStatements">>(
      `/api/datasets/${datasetId}/reviews/${reviewId}/statements?${["dataset", "relationship"].includes(section) ? `section=${section}` : `column_id=${section}`}&offset=${offset}`,
    ),
  conversations: (datasetId: string) =>
    call<import("./generated/schema").components["schemas"]["ConversationView"][]>(`/api/datasets/${datasetId}/conversations`),
  createConversation: (datasetId: string, review_id: string, generation: number) =>
    call<import("./generated/schema").components["schemas"]["ConversationView"]>(
      `/api/datasets/${datasetId}/conversations`,
      json("POST", { review_id, generation }),
    ),
  runFiles: (runId: string) => call<RunFiles>(`/api/runs/${encodeURIComponent(runId)}`),
  fileUrl: (runId: string, name: string, raw = false) => `/api/runs/${encodeURIComponent(runId)}/files/${encodeURIComponent(name)}${raw ? "?raw=1" : ""}`,
  fileText: async (runId: string, name: string, raw = false) => {
    const r = await fetch(api.fileUrl(runId, name, raw));
    if (!r.ok) throw new ApiError(r.status, r.statusText);
    return r.text();
  },
};
