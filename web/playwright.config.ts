import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
process.loadEnvFile(resolve(import.meta.dirname, "../.env"));
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:5178",
    viewport: { width: 1280, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
});
