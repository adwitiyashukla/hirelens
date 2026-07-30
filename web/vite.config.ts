import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API runs on a different port in development. Proxying keeps the
    // frontend origin-relative, so the same build works in production where
    // FastAPI serves these files itself and no proxy exists.
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    // FastAPI serves this directory as static files in the container image.
    outDir: "dist",
    sourcemap: true,
  },
});
