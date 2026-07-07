import { defineConfig, devices } from "@playwright/test";

/**
 * Two projects:
 *  - `seed` (default): app runs with DATABASE_URL unset → pure committed-seed-JSON
 *    mode (deterministic reads), WT calls routed to the local fixture server.
 *  - `db` (opt-in): app runs against a Neon TEST branch (TEST_DATABASE_URL) for
 *    write flows (picks, account claim). Guarded by a URL sentinel in globalSetup.
 */
const PORT = 4028;
const BASE = `http://127.0.0.1:${PORT}`;
const WT_PORT = 4999;

const useDb = process.env.PW_PROJECT === "db";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  globalSetup: "./e2e/helpers/global-setup.ts",
  use: {
    baseURL: BASE,
    trace: "retain-on-failure",
    ...devices["Pixel 7"],
    contextOptions: { reducedMotion: "reduce" },
  },
  projects: [
    { name: "seed", testIgnore: /.*\.db\.spec\.ts/ },
    { name: "db", testMatch: /.*\.db\.spec\.ts/ },
  ],
  webServer: {
    command: "npm run build && npm run start -- -p " + PORT,
    url: BASE,
    reuseExistingServer: false,
    timeout: 180_000,
    env: {
      DATABASE_URL: useDb ? (process.env.TEST_DATABASE_URL ?? "") : "",
      WT_API_BASE_OVERRIDE: `http://127.0.0.1:${WT_PORT}`,
      CRON_SECRET: "e2e-secret",
      NODE_ENV: "production",
    },
  },
});
