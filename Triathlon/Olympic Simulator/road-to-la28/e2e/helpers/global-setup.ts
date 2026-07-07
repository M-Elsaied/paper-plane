/**
 * Starts the WT fixture server for the duration of the E2E run. Returns a
 * teardown that stops it. (Playwright globalSetup can return a teardown fn.)
 */
import { spawn, type ChildProcess } from "node:child_process";
import { join } from "node:path";

let server: ChildProcess | undefined;

export default async function globalSetup() {
  const script = join(process.cwd(), "e2e", "helpers", "wt-fixture-server.mjs");
  server = spawn(process.execPath, [script], { stdio: "inherit", env: { ...process.env, WT_FIXTURE_PORT: "4999" } });
  // Give it a moment to bind.
  await new Promise((r) => setTimeout(r, 500));
  return async () => {
    server?.kill();
  };
}
