// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import DatasetProfile from "./DatasetProfile";
import { api } from "../api";
import type { ProfileView, SavedColumn } from "../types";
vi.mock("../api", () => ({ api: { profile: vi.fn(), generateProfile: vi.fn(), profileColumn: vi.fn() } }));
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
const absent: ProfileView = { status: "not_profiled", profile: null };
const report: ProfileView = {
  status: "partial",
  profile: {
    profile_id: "p",
    dataset_id: "a",
    version_id: "v",
    source_checksum: "sha",
    profiler_version: "1",
    settings_fingerprint: "settings",
    created_at: "2026-01-01",
    updated_at: "2026-01-01",
    status: "partial",
    dataset: {
      rows: 2,
      columns: 1,
      duplicate_rows: 0,
      candidate_keys: [[0]],
      key_search_limit: "Bounded search",
      checks: [{ name: "structure", status: "computed", reason: null }],
      issues: [],
    },
    columns: [{ column_id: "c", position: 0, name: "amount", key: "amount", status: "partial", suggested_dtype: "mixed", issue_count: 1 }],
  },
};
const column: SavedColumn = {
  profile_id: "p",
  version_id: "v",
  column_id: "c",
  source_checksum: "sha",
  result: {
    position: 0,
    name: "amount",
    key: "amount",
    status: "partial",
    empty_count: 0,
    whitespace_only_count: 0,
    distinct_count: 2,
    suggested_dtype: "mixed",
    numeric_parse_count: 1,
    numeric_exception_count: 1,
    numeric: null,
    top_values: [],
    suspected_sentinels: [],
    sentinel_values_truncated: false,
    checks: [{ name: "numeric", status: "unavailable", reason: "One value could not be interpreted." }],
    issues: [
      {
        id: "issue",
        check: "dtype",
        code: "numeric_exceptions",
        reason: "No values discarded",
        column_positions: [0],
        affected_count: 1,
        examples: [" <b>NA</b> "],
        examples_truncated: false,
      },
    ],
  },
};

it("loads without generating and shows a bounded profiling action", async () => {
  let resolve!: (v: ProfileView) => void;
  vi.mocked(api.profile).mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  render(<DatasetProfile datasetId="a" />);
  expect(screen.getByText("Loading saved profile…")).toBeTruthy();
  await act(async () => resolve(absent));
  expect(screen.getByText("No profile yet.")).toBeTruthy();
  expect(api.generateProfile).not.toHaveBeenCalled();
});

it("prevents repeated submission, shows progress and fetches columns only on expansion", async () => {
  vi.mocked(api.profile).mockResolvedValue(absent);
  let resolve!: (v: ProfileView) => void;
  vi.mocked(api.generateProfile).mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  vi.mocked(api.profileColumn).mockResolvedValue(column);
  render(<DatasetProfile datasetId="a" />);
  const button = await screen.findByText("Profile dataset");
  fireEvent.click(button);
  fireEvent.click(button);
  expect(api.generateProfile).toHaveBeenCalledTimes(1);
  expect(screen.getByText("Profiling the original CSV…")).toBeTruthy();
  await act(async () => resolve(report));
  expect(api.profileColumn).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "amount" }));
  const example = await screen.findByText(/<b>NA/);
  expect(example.textContent).toBe(" <b>NA</b> ");
  expect(example.querySelector("b")).toBeNull();
  expect(api.profileColumn).toHaveBeenCalledWith("a", "p", "c");
  expect(screen.getByText(/One value could not/)).toBeTruthy();
});

it("retains a saved report on retry failure and can reload", async () => {
  vi.mocked(api.profile).mockResolvedValue(report);
  vi.mocked(api.generateProfile).mockRejectedValue(new Error("Storage unavailable"));
  render(<DatasetProfile datasetId="a" />);
  fireEvent.click(await screen.findByText("Retry profiling"));
  await screen.findByRole("alert");
  expect(screen.getByText("Rows")).toBeTruthy();
  fireEvent.click(screen.getByText("Reload profile"));
  await screen.findByText("Retry profiling");
  expect(api.profile).toHaveBeenCalledTimes(2);
});

it("retries column errors without regenerating", async () => {
  vi.mocked(api.profile).mockResolvedValue(report);
  vi.mocked(api.profileColumn).mockRejectedValueOnce(new Error("Read failed")).mockResolvedValue(column);
  render(<DatasetProfile datasetId="a" />);
  fireEvent.click(await screen.findByRole("button", { name: "amount" }));
  fireEvent.click(await screen.findByText("Retry column"));
  await screen.findByText(/<b>NA/);
  expect(api.generateProfile).not.toHaveBeenCalled();
});

it("ignores a delayed profile response from the previous route", async () => {
  let resolve!: (v: ProfileView) => void;
  vi.mocked(api.profile)
    .mockImplementationOnce(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    )
    .mockResolvedValue(absent);
  const view = render(<DatasetProfile key="a" datasetId="a" />);
  view.rerender(<DatasetProfile key="b" datasetId="b" />);
  await screen.findByText("No profile yet.");
  await act(async () => resolve(report));
  expect(screen.queryByText("Rows")).toBeNull();
});

it("does not display not-profiled after an initial read failure", async () => {
  vi.mocked(api.profile).mockRejectedValueOnce(new Error("Database unavailable")).mockResolvedValue(absent);
  render(<DatasetProfile datasetId="a" />);
  await screen.findByRole("alert");
  expect(screen.queryByText("No profile yet.")).toBeNull();
  fireEvent.click(screen.getByText("Reload profile"));
  await screen.findByText("No profile yet.");
});

it("ignores generation completion after switching datasets", async () => {
  vi.mocked(api.profile).mockResolvedValue(absent);
  let resolve!: (v: ProfileView) => void;
  vi.mocked(api.generateProfile).mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  const view = render(<DatasetProfile key="a" datasetId="a" />);
  fireEvent.click(await screen.findByText("Profile dataset"));
  view.rerender(<DatasetProfile key="b" datasetId="b" />);
  await screen.findByText("No profile yet.");
  await act(async () => resolve(report));
  expect(screen.queryByText("Rows")).toBeNull();
});

it("does not display a delayed column after it is closed", async () => {
  vi.mocked(api.profile).mockResolvedValue(report);
  let resolve!: (v: SavedColumn) => void;
  vi.mocked(api.profileColumn).mockImplementation(
    () =>
      new Promise((r) => {
        resolve = r;
      }),
  );
  render(<DatasetProfile datasetId="a" />);
  const button = await screen.findByRole("button", { name: "amount" });
  fireEvent.click(button);
  expect(screen.getByText("Loading column…")).toBeTruthy();
  fireEvent.click(button);
  await act(async () => resolve(column));
  expect(screen.queryByText(/<b>NA/)).toBeNull();
});
