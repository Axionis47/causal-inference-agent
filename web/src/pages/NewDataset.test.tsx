// @vitest-environment jsdom
import { afterEach, expect, it, vi } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import NewDataset from "./NewDataset";
import { api } from "../api";

vi.mock("../components/Shell", () => ({ useDatasets: () => ({ refresh: vi.fn() }) }));
vi.mock("../api", () => ({ api: { ingest: vi.fn(), dataset: vi.fn() } }));
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("preserves the form on failure, reuses the submission key, and confirms without starting chat", async () => {
  const ingest = vi.mocked(api.ingest);
  ingest.mockRejectedValueOnce(new Error("Storage unavailable"));
  ingest.mockResolvedValueOnce({
    dataset_id: "9a391312-dd41-462b-babc-99812685a401",
    version_id: "a0aa6ba5-6d46-40a3-8d1d-244c39112751",
    title: "Study",
    filename: "study.csv",
    size_bytes: 9,
    sha256: "0".repeat(64),
    initial_context: " original\naccount ",
    created_at: "2026-01-01",
    status: "saved",
  });
  render(
    <MemoryRouter>
      <Routes>
        <Route path="/" element={<NewDataset />} />
        <Route path="/datasets/:id" element={<p>Saved detail route</p>} />
      </Routes>
    </MemoryRouter>,
  );
  fireEvent.change(screen.getByLabelText("Dataset title"), { target: { value: "Study" } });
  fireEvent.change(screen.getByLabelText("Original study context (optional)"), { target: { value: " original\naccount " } });
  fireEvent.change(screen.getByLabelText("CSV file"), { target: { files: [new File(["a,b\n1,2\n"], "study.csv")] } });
  fireEvent.click(screen.getByText("Save dataset"));
  await screen.findByText("Storage unavailable");
  expect((screen.getByLabelText("Dataset title") as HTMLInputElement).value).toBe("Study");
  expect((screen.getByLabelText("Original study context (optional)") as HTMLTextAreaElement).value).toBe(" original\naccount ");
  fireEvent.click(screen.getByText("Save dataset"));
  await screen.findByText("Saved detail route");
  expect(ingest.mock.calls[0][0].name).toBe("study.csv");
  expect(ingest.mock.calls[0][0].size).toBe(8);
  expect(ingest.mock.calls[0][1]).toBe("Study");
  expect(ingest.mock.calls[0][2]).toBe(" original\naccount ");
  expect(ingest.mock.calls[0][3]).toMatch(/^[0-9a-f-]{36}$/);
  expect(ingest.mock.calls[1].slice(0, 4)).toEqual(ingest.mock.calls[0].slice(0, 4));
});

it("disables repeated submission and shows upload progress", async () => {
  vi.mocked(api.ingest).mockImplementation((_file, _title, _context, _key, progress) => {
    progress(40);
    return new Promise(() => {});
  });
  render(
    <MemoryRouter>
      <NewDataset />
    </MemoryRouter>,
  );
  fireEvent.change(screen.getByLabelText("Dataset title"), { target: { value: "Study" } });
  fireEvent.change(screen.getByLabelText("CSV file"), { target: { files: [new File(["a\n1"], "x.csv")] } });
  fireEvent.click(screen.getByText("Save dataset"));
  await waitFor(() => expect(screen.getByText("Uploading: 40%")).toBeTruthy());
  expect((screen.getByText("Saving…") as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByText("Saving…"));
  expect(api.ingest).toHaveBeenCalledTimes(1);
});

it("changes submission identity when a failed form is edited", async () => {
  const ingest = vi.mocked(api.ingest).mockRejectedValue(new Error("Retry"));
  render(
    <MemoryRouter>
      <NewDataset />
    </MemoryRouter>,
  );
  fireEvent.change(screen.getByLabelText("Dataset title"), { target: { value: "Study" } });
  fireEvent.change(screen.getByLabelText("CSV file"), { target: { files: [new File(["a\n1"], "x.csv")] } });
  fireEvent.click(screen.getByText("Save dataset"));
  await screen.findByText("Retry");
  fireEvent.change(screen.getByLabelText("Original study context (optional)"), { target: { value: "Different" } });
  fireEvent.click(screen.getByText("Save dataset"));
  await screen.findByText("Retry");
  expect(ingest.mock.calls[1][3]).not.toBe(ingest.mock.calls[0][3]);
  expect(ingest.mock.calls[1][2]).toBe("Different");
});
