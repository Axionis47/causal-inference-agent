import { defineConfig } from "@playwright/test";

if (!process.env.INGESTION_BROWSER_URL) throw new Error("Use make container-smoke; a disposable app URL is required");
export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  retries: 0,
  use: { baseURL: process.env.INGESTION_BROWSER_URL, browserName: "chromium", trace: "retain-on-failure" },
  reporter: [["list"]],
});
