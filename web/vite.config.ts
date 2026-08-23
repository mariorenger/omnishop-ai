import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the FastAPI backend so the SPA is same-origin in
// dev; in production nginx does the same reverse-proxy (see nginx.conf).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: { "/api": "http://localhost:8000" },
  },
});
