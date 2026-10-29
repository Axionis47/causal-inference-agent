// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";

class Transport {
  static current: Transport;
  status = 0;
  responseText = "";
  timeout = 0;
  body?: FormData;
  upload = { onprogress: (_: { lengthComputable: boolean; loaded: number; total: number }) => {} };
  onload = () => {};
  onerror = () => {};
  ontimeout = () => {};
  open = vi.fn();
  constructor() {
    Transport.current = this;
  }
  send(body: FormData) {
    this.body = body;
  }
}
beforeEach(() => vi.stubGlobal("XMLHttpRequest", Transport));
afterEach(() => vi.unstubAllGlobals());

it("uses the real adapter to transmit exact multipart fields and bytes", async () => {
  const file = new File(["a,b\n1,2\n"], "study.csv", { type: "text/csv" });
  const progress = vi.fn();
  const pending = api.ingest(file, "Study", " original\naccount ", "request-1", progress);
  const xhr = Transport.current;
  expect(xhr.open).toHaveBeenCalledWith("POST", "/api/datasets");
  expect([...xhr.body!.keys()]).toEqual(["file", "title", "context", "idempotency_key"]);
  expect(xhr.body!.get("title")).toBe("Study");
  expect(xhr.body!.get("context")).toBe(JSON.stringify(" original\naccount "));
  expect(xhr.body!.get("idempotency_key")).toBe("request-1");
  const sent = xhr.body!.get("file") as File;
  expect(sent.name).toBe(file.name);
  const text = await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsText(sent);
  });
  expect(text).toBe("a,b\n1,2\n");
  xhr.upload.onprogress({ lengthComputable: true, loaded: 4, total: 10 });
  expect(progress).toHaveBeenCalledWith(40);
  xhr.upload.onprogress({ lengthComputable: false, loaded: 6, total: 0 });
  expect(progress).toHaveBeenCalledTimes(1);
  xhr.status = 201;
  xhr.responseText = JSON.stringify({ status: "saved" });
  xhr.onload();
  await expect(pending).resolves.toEqual({ status: "saved" });
});

it.each([
  [422, JSON.stringify({ detail: "Invalid CSV" }), "Invalid CSV"],
  [503, "proxy failure", "Unexpected server response"],
])("handles HTTP %s without a false success", async (status, response, message) => {
  const pending = api.ingest(new File(["a\n1"], "x.csv"), "Study", "", "key", vi.fn());
  const xhr = Transport.current;
  xhr.status = status;
  xhr.responseText = response;
  xhr.onload();
  await expect(pending).rejects.toThrow(message);
});

it.each(["onerror", "ontimeout"] as const)("makes %s retryable", async (event) => {
  const pending = api.ingest(new File(["a\n1"], "x.csv"), "Study", "", "key", vi.fn());
  expect(Transport.current.timeout).toBe(180000);
  Transport.current[event]();
  await expect(pending).rejects.toThrow(/Retry the same submission/);
});
