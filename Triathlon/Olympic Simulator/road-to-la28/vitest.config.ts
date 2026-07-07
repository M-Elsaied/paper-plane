import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

// Base config (shared resolve alias). Projects are defined in vitest.workspace.ts.
export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      // `server-only` throws outside an RSC — stub it so server modules load.
      "server-only": fileURLToPath(new URL("./tests/helpers/empty.ts", import.meta.url)),
    },
  },
  test: {
    environment: "node",
  },
});
