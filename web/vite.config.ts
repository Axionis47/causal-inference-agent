import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: `http://127.0.0.1:${loadEnv(mode, ".", "API_PORT").API_PORT ?? "8000"}`, changeOrigin: true } },
  },
  build: { outDir: "dist", emptyOutDir: true },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
}));
