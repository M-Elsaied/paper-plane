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
  "/events/195148": "hamburg-event.json",
  "/events/195148/programs": "hamburg-programs.json",
  "/events/195148/programs/678086/entries": "hamburg-men-entries.json",
  "/events/195145/programs": "yokohama-programs.json",
  "/events/195145/programs/677500/results": "yokohama-men-results.json",
  "/rankings/11": "ranking-11-oqr-men.json",
  "/rankings/12": "ranking-12-oqr-women.json",
  "/rankings/15": "ranking-15-wtcs-men.json",
  "/rankings/16": "ranking-16-wtcs-women.json",
  "/rankings/64": "ranking-64-mr.json",
  "/athletes/70338/results": "athlete-70338-results.json",
  "/athletes/70338": "athlete-70338.json",
};

const PORT = Number(process.env.WT_FIXTURE_PORT || 4999);

// Deterministic athlete-search results (any query) so the live-search picker
// path is exercisable in E2E: one ranked (Vilaça) + one unranked (Elsaied).
const SEARCH_RESULTS = {
  code: 200,
  status: "success",
  data: [
    { athlete_id: 86042, athlete_title: "Vasco Vilaca", athlete_noc: "POR", athlete_gender: "male", athlete_yob: "1999" },
    { athlete_id: 70338, athlete_title: "Mohamed Elsaied", athlete_noc: "EGY", athlete_gender: "male", athlete_yob: "1993" },
  ],
};

// Synthetic athlete profiles + results so the Head-to-head (versus) page renders
// in E2E for the two seeded ranked men (86042 Vilaça, 49390 Coninx).
const VS = {
  86042: { noc: "POR", name: "Vasco Vilaca", stats: { race_starts: 90, race_wins: 12, race_podiums: 31, finish_percentage: 95 } },
  49390: { noc: "FRA", name: "Dorian Coninx", stats: { race_starts: 117, race_wins: 21, race_podiums: 38, finish_percentage: 92 } },
};
function vsProfile(id) {
  const a = VS[id];
  return { code: 200, status: "success", data: { athlete_id: Number(id), athlete_title: a.name, athlete_noc: a.noc, athlete_gender: "male", athlete_yob: "1998", stats: a.stats, latest_results: [] } };
}
function vsResults(id) {
  // Two common events (1,2) with opposite winners → 1-1 in the shared set.
  const pos = id == 86042 ? [1, 2] : [4, 1];
  return { code: 200, status: "success", data: [
    { event_id: 1, event_title: "2026 WTCS Alghero", event_date: "2026-05-30", prog_name: "Elite Men", position: pos[0] },
    { event_id: 2, event_title: "2026 WTCS Hamburg", event_date: "2026-06-20", prog_name: "Elite Men", position: pos[1] },
  ] };
}

createServer((req, res) => {
  const path = (req.url || "").split("?")[0];
  res.setHeader("content-type", "application/json");
  if (path === "/search/athletes") {
    res.writeHead(200);
    res.end(JSON.stringify(SEARCH_RESULTS));
    return;
  }
  let m = path.match(/^\/athletes\/(86042|49390)(\/results)?$/);
  if (m) {
    res.writeHead(200);
    res.end(JSON.stringify(m[2] ? vsResults(m[1]) : vsProfile(m[1])));
    return;
  }
  const file = MAP[path];
  if (file && existsSync(join(FIX, file))) {
    res.writeHead(200);
    res.end(readFileSync(join(FIX, file)));
  } else {
    // Empty-but-valid envelope so unmapped calls degrade gracefully (fallbacks).
    res.writeHead(200);
    res.end(JSON.stringify({ code: 200, status: "success", data: [] }));
  }
}).listen(PORT, "127.0.0.1", () => console.log(`WT fixture server on http://127.0.0.1:${PORT}`));
