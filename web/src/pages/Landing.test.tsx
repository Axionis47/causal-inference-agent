// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";
import { cleanup, render, screen, fireEvent, act, within } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import Shell from "../components/Shell";
import Landing from "./Landing";
import { api } from "../api";
import type { DatasetList, DatasetListItem } from "../types";
vi.mock("../api", () => ({ api: { listDatasets: vi.fn() } }));
vi.mock("../hooks/useNarrow", () => ({ useNarrow: () => false }));
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
const item = (id: string): DatasetListItem => ({
  dataset_id: id,
  version_id: id,
  title: id,
  filename: "x.csv",
  size_bytes: 4,
  created_at: "2026-01-01",
  status: "saved",
  has_initial_context: false,
});
function view() {
  render(
    <MemoryRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<Landing />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}
const main = () => within(document.querySelector(".main")! as HTMLElement);
it("distinguishes loading, failed listing and a genuinely empty library", async () => {
  let reject!: (e: Error) => void;
  vi.mocked(api.listDatasets)
    .mockImplementationOnce(
      () =>
        new Promise((_r, r) => {
          reject = r;
        }),
    )
    .mockResolvedValue({ items: [], next_cursor: null });
  view();
  expect(main().getByRole("status").textContent).toContain("Loading");
  await act(async () => reject(new Error("Storage offline")));
  expect(main().queryByText(/No datasets yet/)).toBeNull();
  fireEvent.click(main().getByText("Retry"));
  await main().findByText(/No datasets yet/);
});
it("shares loaded items and preserves them when the next page fails", async () => {
  vi.mocked(api.listDatasets)
    .mockResolvedValueOnce({ items: [item("First")], next_cursor: "cursor" })
    .mockRejectedValueOnce(new Error("Retry page"))
    .mockResolvedValueOnce({ items: [item("Second")], next_cursor: null });
  view();
  await main().findByText("First");
  fireEvent.click(main().getByText("Load more"));
  await main().findByText("Retry page");
  expect(main().getByText("First")).toBeTruthy();
  fireEvent.click(main().getByText("Retry"));
  await main().findByText("Second");
  expect(api.listDatasets).toHaveBeenLastCalledWith("cursor");
  expect(within(screen.getByRole("navigation")).getByText("Second")).toBeTruthy();
  expect(main().getByText("Second").getAttribute("href")).toBe("/datasets/Second");
});
it("a refresh invalidates a delayed next page", async () => {
  let resolve!: (page: DatasetList) => void;
  vi.mocked(api.listDatasets)
    .mockResolvedValueOnce({ items: [item("First")], next_cursor: "cursor" })
    .mockImplementationOnce(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    )
    .mockResolvedValueOnce({ items: [item("Fresh")], next_cursor: null });
  // Exercise the shared hook directly to trigger refresh while pagination is pending.
  const { useDatasetLibrary } = await import("../hooks/useDatasetLibrary");
  function Harness() {
    const lib = useDatasetLibrary();
    return (
      <>
        <button onClick={() => void lib.refresh()}>refresh</button>
        <button onClick={() => void lib.loadMore()}>more</button>
        <p>{lib.items?.map((i) => i.title).join(",")}</p>
      </>
    );
  }
  render(<Harness />);
  await screen.findByText("First");
  fireEvent.click(screen.getByText("more"));
  fireEvent.click(screen.getByText("refresh"));
  await screen.findByText("Fresh");
  await act(async () => resolve({ items: [item("Stale")], next_cursor: null }));
  expect(screen.queryByText(/Stale/)).toBeNull();
});
