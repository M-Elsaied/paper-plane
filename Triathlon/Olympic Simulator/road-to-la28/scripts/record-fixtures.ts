/**
 * Record World Triathlon API fixtures for tests. Saves the FULL response
 * envelope (what wtGet returns) so the test fetch-stub can replay it verbatim.
 * Polite: a handful of requests. Run:  npm run fixtures:record [-- --force]
 */
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { wtGet } from "../src/lib/wt-api/client";

const OUT = join(process.cwd(), "tests", "fixtures", "wt");
const force = process.argv.includes("--force");

interface Target {
  file: string;
  path: string;
  params?: Record<string, string | number>;
}

const TARGETS: Target[] = [
  { file: "ranking-11-oqr-men.json", path: "/rankings/11", params: { limit: 1000 } },
  { file: "ranking-12-oqr-women.json", path: "/rankings/12", params: { limit: 1000 } },
  { file: "ranking-15-wtcs-men.json", path: "/rankings/15", params: { limit: 1000 } },
  { file: "ranking-16-wtcs-women.json", path: "/rankings/16", params: { limit: 1000 } },
  { file: "ranking-64-mr.json", path: "/rankings/64", params: { limit: 100 } },
  { file: "yokohama-programs.json", path: "/events/195145/programs" },
  { file: "yokohama-men-results.json", path: "/events/195145/programs/677500/results" },
  { file: "hamburg-programs.json", path: "/events/195148/programs" },
  { file: "hamburg-men-entries.json", path: "/events/195148/programs/678086/entries" },
  // An unranked athlete (Mohamed Elsaied, EGY) for the profile-view path.
  { file: "athlete-70338.json", path: "/athletes/70338" },
  { file: "athlete-70338-results.json", path: "/athletes/70338/results", params: { per_page: 8 } },
];

async function main() {
  mkdirSync(OUT, { recursive: true });
  const meta: Record<string, { hash: string; recordedAt: string }> = {};
  const recordedAt = new Date().toISOString();

  for (const t of TARGETS) {
    const dest = join(OUT, t.file);
    if (existsSync(dest) && !force) {
      console.log("skip (exists, use --force):", t.file);
      continue;
    }
    const res = await wtGet(t.path, t.params ?? {});
    const json = JSON.stringify(res, null, 0);
    writeFileSync(dest, JSON.stringify(res, null, 2));
    meta[t.file] = { hash: createHash("sha256").update(json).digest("hex").slice(0, 16), recordedAt };
    console.log("wrote", t.file);
  }

  writeFileSync(join(OUT, "fixtures-meta.json"), JSON.stringify({ recordedAt, files: meta }, null, 2));
  console.log("✓ fixtures recorded to tests/fixtures/wt/");
}

main().catch((e) => {
  console.error("record failed:", e);
  process.exit(1);
});
