import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { WtRoute } from "./mock-wt";

/** Load a recorded WT fixture (the full response envelope). */
export function loadFixture<T = unknown>(name: string): T {
  return JSON.parse(readFileSync(join(process.cwd(), "tests/fixtures/wt", name), "utf8")) as T;
}

/** Standard fetch routes covering the 5 rankings the sync cron fetches. */
export function rankingRoutes(): WtRoute[] {
  return [
    { match: "/rankings/11", response: loadFixture("ranking-11-oqr-men.json") },
    { match: "/rankings/12", response: loadFixture("ranking-12-oqr-women.json") },
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
