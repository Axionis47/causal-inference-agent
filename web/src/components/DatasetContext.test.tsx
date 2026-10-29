// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import DatasetContext from "./DatasetContext";
import { api } from "../api";
import type { ContextView, ContextDatasetView, ContextColumnView } from "../types";
vi.mock("../api", () => ({ api: { context: vi.fn(), prepareContext: vi.fn(), datasetContext: vi.fn(), columnContext: vi.fn() } }));
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
const absent: ContextView = { status: "not_prepared", pack: null };
const source = "😀 amount is measured in dollars.\n<b>literal</b>";
const cite = { source_line: 1, start: 2, end: 32, quote: "amount is measured in dollars." };
const columns = [
  { column_id: "c1", position: 0, name: "code", key: "code", statement_count: 0, relationship_count: 1 },
  { column_id: "c2", position: 1, name: "amount", key: "amount", statement_count: 1, relationship_count: 1 },
];
const pack: NonNullable<ContextView["pack"]> = {
  revision_id: "p",
  dataset_id: "a",
  version_id: "v",
  revision: 1,
  created_at: "2026-01-01",
  source_id: "s",
  source_checksum: "sha",
  csv_checksum: "csv",
  column_index_fingerprint: "cols",
  schema_version: "1",
  extractor_version: "1",
  model: "test",
  source_present: true,
  columns,
};
const saved: ContextView = { status: "prepared", pack };
const dataset: ContextDatasetView = { pack, source_text: source, dataset: [], concepts: [], relationships: [] };
const statementDefaults = {
  column_ids: [],
  participants: [],
  timing: "",
  conditions: "",
  directness: "unspecified" as const,
  absence_scope: "unspecified" as const,
  status: "proposed" as const,
  review_source: null,
  replaces: null,
  reason: "",
  binding: null,
};
const col: ContextColumnView = {
  revision_id: "p",
  version_id: "v",
  column: { column_id: "c2", position: 1, name: "amount", key: "amount" },
  statements: [
    {
      ...statementDefaults,
      id: "s1",
      scope: "column",
      source: { kind: "extraction", id: "original", text: "" },
      field: "units",
      value: "dollars",
      basis: "source_statement",
      citations: [cite],
      qualifications: "",
    },
  ],
  dataset: [],
  relationships: [],
  concepts: [],
};
function setup() {
  vi.mocked(api.context).mockResolvedValue(saved);
  vi.mocked(api.datasetContext).mockResolvedValue(dataset);
  vi.mocked(api.columnContext).mockResolvedValue(col);
}

it("loads without generation and fetches column context only on expansion", async () => {
  setup();
  vi.mocked(api.columnContext).mockResolvedValue({ ...col, statements: [{ ...col.statements[0], value: "<b>dollars</b>" }] });
  render(<DatasetContext datasetId="a" />);
  expect(screen.getByText("Loading saved context…")).toBeTruthy();
  fireEvent.click(await screen.findByText("View context"));
  await screen.findByRole("heading", { name: "Columns" });
  expect(api.prepareContext).not.toHaveBeenCalled();
  expect(api.columnContext).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "amount" }));
  const value = await screen.findByText("<b>dollars</b>");
  expect(value.querySelector("b")).toBeNull();
  expect(api.columnContext).toHaveBeenCalledWith("a", "p", "c2");
  expect(screen.getAllByText("Not supplied").length).toBeGreaterThan(0);
  fireEvent.click(screen.getByText("View source 1"));
  const passage = screen.getByLabelText("Original source passage");
  expect(passage.querySelector("p")?.textContent).toBe(source);
  expect(passage.querySelector("b")).toBeNull();
  expect(passage.querySelector("mark")?.textContent).toBe("amount is measured in dollars.");
});

it("prevents duplicate preparation and shows progress", async () => {
  setup();
  vi.mocked(api.context).mockResolvedValue(absent);
  let resolve!: (value: ContextView) => void;
  vi.mocked(api.prepareContext).mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  render(<DatasetContext datasetId="a" />);
  const button = await screen.findByText("Prepare context");
  fireEvent.click(button);
  fireEvent.click(button);
  expect(api.prepareContext).toHaveBeenCalledTimes(1);
  expect(screen.getByText("Arranging the supplied context…")).toBeTruthy();
  await act(async () => resolve(saved));
  await screen.findByRole("heading", { name: "Organized context" });
});

it("keeps saved context after a refresh failure and retries without extraction", async () => {
  setup();
  vi.mocked(api.context).mockResolvedValueOnce(saved).mockRejectedValueOnce(new Error("Unavailable")).mockResolvedValue(saved);
  render(<DatasetContext datasetId="a" />);
  fireEvent.click(await screen.findByText("Refresh context"));
  expect(await screen.findByRole("alert")).toBeTruthy();
  expect(screen.getByText("View context")).toBeTruthy();
  fireEvent.click(screen.getByText("Refresh context"));
  fireEvent.click(screen.getByText("View context"));
  await screen.findByRole("heading", { name: "Organized context" });
  expect(api.prepareContext).not.toHaveBeenCalled();
});

it("retries initial and column reads without generating", async () => {
  setup();
  vi.mocked(api.context).mockRejectedValueOnce(new Error("Offline")).mockResolvedValue(saved);
  vi.mocked(api.columnContext).mockRejectedValueOnce(new Error("Column unavailable")).mockResolvedValue(col);
  render(<DatasetContext datasetId="a" />);
  fireEvent.click(await screen.findByText("Reload context"));
  fireEvent.click(await screen.findByText("View context"));
  fireEvent.click(await screen.findByRole("button", { name: "amount" }));
  fireEvent.click(await screen.findByText("Retry column context"));
  await screen.findByText("dollars");
  expect(api.prepareContext).not.toHaveBeenCalled();
});

it("ignores delayed list and preparation responses after changing datasets", async () => {
  setup();
  let resolve!: (v: ContextView) => void;
  vi.mocked(api.context)
    .mockImplementationOnce(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    )
    .mockResolvedValue(absent);
  const view = render(<DatasetContext datasetId="a" />);
  view.rerender(<DatasetContext datasetId="b" />);
  await screen.findByText("Prepare context");
  await act(async () => resolve(saved));
  expect(screen.queryByText("View context")).toBeNull();
  let finish!: (v: ContextView) => void;
  vi.mocked(api.prepareContext).mockImplementation(
    () =>
      new Promise((r) => {
        finish = r;
      }),
  );
  fireEvent.click(screen.getByText("Prepare context"));
  view.rerender(<DatasetContext datasetId="c" />);
  await screen.findByText("Prepare context");
  await act(async () => finish(saved));
  expect(screen.queryByLabelText("Organized context")).toBeNull();
});

it("ignores delayed column responses when another column is opened", async () => {
  setup();
  let resolve!: (v: ContextColumnView) => void;
  vi.mocked(api.columnContext)
    .mockImplementationOnce(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    )
    .mockResolvedValue({ ...col, column: { ...col.column, column_id: "c1", name: "code" }, statements: [] });
  render(<DatasetContext datasetId="a" />);
  fireEvent.click(await screen.findByText("View context"));
  fireEvent.click(await screen.findByRole("button", { name: "amount" }));
  fireEvent.click(screen.getByRole("button", { name: "code" }));
  await screen.findByLabelText("Context for code");
  await act(async () => resolve(col));
  expect(screen.queryByText("dollars")).toBeNull();
});

it("links a shared relationship to the other participating column", async () => {
  setup();
  const relation: ContextColumnView["relationships"][number] = {
    ...statementDefaults,
    id: "r1",
    scope: "relationship",
    field: "unclassified",
    source: { kind: "extraction", id: "original", text: "" },
    value: "Recorded together",
    basis: "interpretation",
    citations: [cite],
    qualifications: "No causal conclusion",
    timing: "same visit",
    conditions: "",
    directness: "unspecified",
    absence_scope: "unspecified",
    participants: [
      { kind: "column", id: "c1", role: "participant" },
      { kind: "column", id: "c2", role: "participant" },
    ],
  };
  vi.mocked(api.columnContext).mockResolvedValue({ ...col, relationships: [relation] });
  render(<DatasetContext datasetId="a" />);
  fireEvent.click(await screen.findByText("View context"));
  fireEvent.click(await screen.findByRole("button", { name: "amount" }));
  const panel = await screen.findByLabelText("Context for amount");
  fireEvent.click(within(panel).getByRole("button", { name: "code" }));
  await act(async () => {});
  expect(api.columnContext).toHaveBeenLastCalledWith("a", "p", "c1");
});
