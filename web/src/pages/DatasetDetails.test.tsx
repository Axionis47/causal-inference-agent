// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";
import { cleanup, render, screen, fireEvent, act } from "@testing-library/react";
import { MemoryRouter, Route, Routes, Link } from "react-router-dom";
import DatasetDetails from "./DatasetDetails";
import { api } from "../api";
import type { IngestedDataset } from "../types";
vi.mock("../api", () => ({
  api: {
    dataset: vi.fn(),
    conversations: vi.fn().mockResolvedValue([]),
    startReview: vi.fn(),
    context: vi.fn().mockResolvedValue({ status: "not_prepared", pack: null }),
    profile: vi.fn().mockResolvedValue({ status: "not_profiled", profile: null }),
  },
}));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
const record = (id: string): IngestedDataset => ({
  dataset_id: id,
  version_id: id,
  title: id,
  filename: "a.csv",
  size_bytes: 4,
  sha256: "0".repeat(64),
  initial_context: " <b>literal</b>\noriginal ",
  created_at: "2026-01-01",
  status: "saved",
});
function view() {
  return render(
    <MemoryRouter initialEntries={["/datasets/a"]}>
      <Link to="/datasets/b">Next dataset</Link>
      <Routes>
        <Route path="/datasets/:datasetId" element={<DatasetDetails />} />
        <Route path="/d/:name" element={<h1>Conversation</h1>} />
        <Route path="/datasets/:datasetId/reviews/:reviewId" element={<h1>Review context</h1>} />
      </Routes>
    </MemoryRouter>,
  );
}
it("loads saved context as exact text and offers a download", async () => {
  vi.mocked(api.dataset).mockResolvedValue(record("a"));
  view();
  await screen.findByRole("heading", { name: "a" });
  const context = screen.getByText(/literal/);
  expect(context.textContent).toBe(record("a").initial_context);
  expect(context.querySelector("b")).toBeNull();
  expect(screen.getByRole("link", { name: "Download original CSV" }).getAttribute("href")).toBe("/api/datasets/a/file");
});
it("retries retrieval errors", async () => {
  vi.mocked(api.dataset).mockRejectedValueOnce(new Error("Unavailable")).mockResolvedValue(record("a"));
  view();
  await screen.findByRole("alert");
  fireEvent.click(screen.getByText("Retry"));
  await screen.findByRole("heading", { name: "a" });
});
it("collects a question before opening the context review", async () => {
  vi.mocked(api.dataset).mockResolvedValue(record("a"));
  vi.mocked(api.startReview).mockResolvedValue({
    id: "review-a",
    dataset_id: "a",
    version_id: "v1",
    question: "What does this dataset mean?",
    starting_revision_id: "r1",
    completed_revision_id: null,
    generation: 1,
    columns: [],
    edits: [],
    dispositions: [],
  });
  view();
  const button = await screen.findByRole("button", { name: "Review context for this question" });
  expect(button.hasAttribute("disabled")).toBe(true);
  fireEvent.change(screen.getByLabelText("What question would you like to explore?"), { target: { value: "What does this dataset mean?" } });
  fireEvent.click(button);
  await screen.findByRole("heading", { name: "Review context" });
  expect(api.startReview).toHaveBeenCalledWith("a", "What does this dataset mean?");
});
it("ignores a delayed response after navigation", async () => {
  let resolve!: (value: IngestedDataset) => void;
  vi.mocked(api.dataset)
    .mockImplementationOnce(
      () =>
        new Promise((r) => {
          resolve = r;
        }),
    )
    .mockResolvedValue(record("b"));
  view();
  fireEvent.click(screen.getByText("Next dataset"));
  await screen.findByRole("heading", { name: "b" });
  await act(async () => resolve(record("a")));
  expect(screen.queryByRole("heading", { name: "a" })).toBeNull();
});
