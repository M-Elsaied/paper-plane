import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { WtRoute } from "./mock-wt";

/** Load a recorded WT fixture (the full response envelope). */
export function loadFixture<T = unknown>(name: string): T {
  return JSON.parse(readFileSync(join(process.cwd(), "tests/fixtures/wt", name), "utf8")) as T;
}

/** A small synthetic World Ranking payload (13/14) — keeps fixtures lean while
 *  exercising the world ingest/read path in the sync cron. */
export function syntheticWorld(rankingId: number, gender: "male" | "female") {
  const nocs = ["FRA", "GBR", "EGY", "RSA", "MAR", "AUS", "JPN", "BRA"];
  return {
    code: 200,
    status: "success",
    data: {
      ranking_id: rankingId,
      ranking_name: gender === "male" ? "Elite Men" : "Elite Women",
      ranking_cat_name: "World Rankings",
      published: "2026-06-28 20:00:00",
      total: nocs.length,
      rankings: nocs.map((noc, i) => ({
        athlete_id: 500000 + rankingId * 100 + i,
        athlete_full_name: `${noc} World-${i + 1}`,
        athlete_noc: noc,
        athlete_gender: gender,
        rank: i + 1,
        total: 1000 - i * 50,
      })),
    },
  };
}

/** Standard fetch routes covering the 7 rankings the sync cron fetches. */
export function rankingRoutes(): WtRoute[] {
  return [
    { match: "/rankings/11", response: loadFixture("ranking-11-oqr-men.json") },
    { match: "/rankings/12", response: loadFixture("ranking-12-oqr-women.json") },
    { match: "/rankings/13", response: syntheticWorld(13, "male") },
    { match: "/rankings/14", response: syntheticWorld(14, "female") },
    { match: "/rankings/15", response: loadFixture("ranking-15-wtcs-men.json") },
    { match: "/rankings/16", response: loadFixture("ranking-16-wtcs-women.json") },
    { match: "/rankings/64", response: loadFixture("ranking-64-mr.json") },
  ];
}

/** Fetch routes for the Yokohama (195145) completed race — Pick-'Em scoring. */
export function yokohamaRoutes(): WtRoute[] {
  return [
    { match: "/events/195145/programs/677500/results", response: loadFixture("yokohama-men-results.json") },
    { match: "/events/195145/programs", response: loadFixture("yokohama-programs.json") },
  ];
}

/** Fetch routes for the Hamburg (195148) upcoming race — start list. */
export function hamburgRoutes(): WtRoute[] {
  return [
    { match: "/events/195148/programs/678086/entries", response: loadFixture("hamburg-men-entries.json") },
    { match: "/events/195148/programs", response: loadFixture("hamburg-programs.json") },
  ];
}
