// The wire, as the server declares it. Every name here is derived from src/generated/schema.ts, which `npm run types`
// generates from the API's OpenAPI schema; nothing is typed by hand. The server always emits every field of a response
// (pydantic fills the defaults), so the optional marks OpenAPI puts on defaulted fields are lifted here.

import type { components } from "./generated/schema";

type DeepRequired<T> = T extends (infer U)[] ? DeepRequired<U>[] : T extends object ? { [K in keyof T]-?: DeepRequired<T[K]> } : T;
type S<K extends keyof components["schemas"]> = DeepRequired<components["schemas"][K]>;

export type NumericShape = S<"NumericShape">;
export type TopValue = S<"TopValue">;
export type DatetimeShape = S<"DatetimeShape">;
export type Sentinel = S<"Sentinel">;
export type ColumnSummary = S<"ColumnSummary">;
export type ProfileOut = S<"ProfileOut">;
export type SessionBrief = S<"SessionBrief">;
export type DatasetSummary = S<"DatasetSummary">;
export type DatasetList = S<"DatasetList">;
export type DatasetCreate = S<"DatasetCreate">;
export type MessageIn = S<"MessageIn">;
export type QuestionView = S<"QuestionView">;
export type ClaimView = S<"ClaimView">;
export type StatusView = S<"StatusView">;
export type CheckView = S<"CheckView">;
export type RefutationView = S<"RefutationView">;
export type InterpretationView = S<"InterpretationView">;
export type EstimateView = S<"EstimateView">;
export type DeclineView = S<"DeclineView">;
export type DecisionView = S<"DecisionView">;
export type FeasibilityView = S<"FeasibilityView">;
export type RunView = S<"RunView">;
export type Turn = S<"Turn">;
export type Prompt = S<"Prompt">;
export type Activity = S<"Activity">;
export type SessionView = S<"SessionView">;
export type Stage = SessionView["stage"];
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
