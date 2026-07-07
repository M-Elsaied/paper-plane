/**
 * Tiny WT API fixture server for E2E. Serves recorded fixtures so the running
 * app (with WT_API_BASE_OVERRIDE pointed here) never touches the live API.
 * Deterministic + offline.  node e2e/helpers/wt-fixture-server.mjs
 */
import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const FIX = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "tests", "fixtures", "wt");

const MAP = {
  "/events/195148/programs": "hamburg-programs.json",
  "/events/195148/programs/678086/entries": "hamburg-men-entries.json",
  "/events/195145/programs": "yokohama-programs.json",
  "/events/195145/programs/677500/results": "yokohama-men-results.json",
  "/rankings/11": "ranking-11-oqr-men.json",
  "/rankings/12": "ranking-12-oqr-women.json",
  "/rankings/15": "ranking-15-wtcs-men.json",
  "/rankings/16": "ranking-16-wtcs-women.json",
  "/rankings/64": "ranking-64-mr.json",
};

const PORT = Number(process.env.WT_FIXTURE_PORT || 4999);

createServer((req, res) => {
  const path = (req.url || "").split("?")[0];
  const file = MAP[path];
  res.setHeader("content-type", "application/json");
  if (file && existsSync(join(FIX, file))) {
    res.writeHead(200);
    res.end(readFileSync(join(FIX, file)));
  } else {
    // Empty-but-valid envelope so unmapped calls degrade gracefully (fallbacks).
    res.writeHead(200);
    res.end(JSON.stringify({ code: 200, status: "success", data: [] }));
  }
}).listen(PORT, "127.0.0.1", () => console.log(`WT fixture server on http://127.0.0.1:${PORT}`));
