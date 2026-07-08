/**
 * Seed script: pull the real World Triathlon OQR + Mixed Relay rankings and
 * write them as committed JSON the app reads without a live database, so the
 * demo runs out of the box. Re-run any time to refresh:  npm run seed
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { RANKING_IDS } from "../src/config/wt-api";
import { fetchRankingState, fetchMrNations, fetchWorldRanking, establishedNocsFrom } from "../src/lib/wt-api/rankings";
import { fetchUpcomingEvents } from "../src/lib/wt-api/events";

const OUT = join(process.cwd(), "src", "data");

/** Seed date is pinned so re-runs are reproducible; matches the app "today". */
const TODAY = "2026-07-06";
function plusDays(iso: string, days: number): string {
  const d = new Date(iso);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  console.log("Fetching OQR men (ranking %d)…", RANKING_IDS.oqr_men.id);
  const men = await fetchRankingState(RANKING_IDS.oqr_men.id, "male");
  console.log("  %d athletes, published %s", men.athletes.length, men.publishedAt);

  console.log("Fetching OQR women (ranking %d)…", RANKING_IDS.oqr_women.id);
  const women = await fetchRankingState(RANKING_IDS.oqr_women.id, "female");
  console.log("  %d athletes, published %s", women.athletes.length, women.publishedAt);

  console.log("Fetching World Ranking men/women (rankings %d/%d)…", RANKING_IDS.world_men.id, RANKING_IDS.world_women.id);
  const worldMen = await fetchWorldRanking(RANKING_IDS.world_men.id, "male", 1000);
  const worldWomen = await fetchWorldRanking(RANKING_IDS.world_women.id, "female", 1000);
  console.log("  %d men, %d women", worldMen.length, worldWomen.length);

  console.log("Fetching Mixed Relay Olympic ranking (ranking %d)…", RANKING_IDS.mr_olympic.id);
  const mr = await fetchMrNations(RANKING_IDS.mr_olympic.id);
  console.log("  %d nations", mr.length);

  console.log("Fetching upcoming elite events (%s → +120d)…", TODAY);
  let events: Awaited<ReturnType<typeof fetchUpcomingEvents>> = [];
  try {
    events = await fetchUpcomingEvents(TODAY, plusDays(TODAY, 120));
    console.log("  %d events", events.length);
  } catch (e) {
    console.warn("  events fetch failed (non-fatal):", (e as Error).message);
  }

  write("qual-state-men.json", men);
  write("qual-state-women.json", women);
  const established = establishedNocsFrom(worldMen, worldWomen);
  console.log("  established (non-New-Flag) nations: %d", established.length);

  write("world-ranking-men.json", worldMen);
  write("world-ranking-women.json", worldWomen);
  write("established-nocs.json", established);
  write("mr-nations.json", mr);
  write("events.json", events);
  write("seed-meta.json", {
    seededAtNote: "run npm run seed to refresh",
    today: TODAY,
    menPublished: men.publishedAt,
    womenPublished: women.publishedAt,
    counts: { men: men.athletes.length, women: women.athletes.length, mrNations: mr.length, events: events.length },
  });
  console.log("✓ wrote seed data to src/data/");
}

function write(name: string, data: unknown) {
  writeFileSync(join(OUT, name), JSON.stringify(data, null, 2));
}

main().catch((err) => {
  console.error("seed failed:", err);
  process.exit(1);
});
