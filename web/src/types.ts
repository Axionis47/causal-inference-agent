// The wire, as the server declares it. Every name here is derived from src/generated/schema.ts, which `npm run types`
// generates from the API's OpenAPI schema; nothing is typed by hand. The server always emits every field of a response
// (pydantic fills the defaults), so the optional marks OpenAPI puts on defaulted fields are lifted here.

import type { components } from "./generated/schema";

type DeepRequired<T> = T extends (infer U)[] ? DeepRequired<U>[] : T extends object ? { [K in keyof T]-?: DeepRequired<T[K]> } : T;
type S<K extends keyof components["schemas"]> = DeepRequired<components["schemas"][K]>;

export type IngestedDataset = S<"IngestedDataset">;
export type DatasetListItem = S<"DatasetListItem">;
export type DatasetList = S<"DatasetList">;
export type CheckView = S<"CheckView">;
export type Falsification = S<"Falsification">;
export type Sensitivity = S<"Sensitivity">;
export type ContextAdmission = S<"ContextAdmission">;
export type InterpretationView = S<"InterpretationView">;
export type EstimateView = S<"EstimateView">;
export type DeclineView = S<"DeclineView">;
export type DecisionView = S<"DecisionView">;
export type FeasibilityView = S<"FeasibilityView">;
export type RunView = S<"RunView">;
export type FileEntry = S<"FileEntry">;
export type RunFiles = S<"RunFiles">;

// figures
export type FigureSpec = S<"FigureSpec">;
export type Series = S<"Series">;
export type Mark = S<"Mark">;
export type GraphNode = S<"Node">;
export type GraphEdge = S<"Edge">;
export type Kind = FigureSpec["kind"];
export type Role = GraphNode["role"];

export type ProfileView = S<"ProfileView">;
export type SavedProfile = S<"SavedProfile">;
export type SavedColumn = S<"SavedColumn">;
export type ProfileCheck = S<"Check">;
export type ProfileIssue = S<"Issue">;

export type ContextView = S<"ContextView">;
export type ContextDatasetView = S<"ContextDatasetView">;
export type ContextColumnView = S<"ContextColumnView">;
export type ContextStatement = S<"KnowledgeStatement">;
export type ContextCitation = S<"Citation">;
