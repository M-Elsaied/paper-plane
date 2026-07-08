import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { installWtFetch } from "../helpers/mock-wt";
import { loadFixture } from "../helpers/fixtures";

// buildVersus pulls buildCockpit (seed JSON) + fetchAthleteProfile/results (WT).
// We stub the athlete endpoints and let cockpit read the committed seed.
import { buildVersus } from "@/lib/versus";

const profile = (id: number, noc: string, stats: { race_starts: number; race_wins: number; race_podiums: number; finish_percentage: number }) => ({
  code: 200,
  status: "ok",
  data: {
    athlete_id: id,
    athlete_title: `Athlete ${id}`,
    athlete_noc: noc,
    athlete_gender: "male",
    athlete_yob: 1998,
    stats,
    latest_results: [],
  },
});

const results = (rows: { event: number; prog: string; pos: number; date: string }[]) => ({
  code: 200,
  status: "ok",
  data: rows.map((r) => ({ event_id: r.event, event_title: `Race ${r.event}`, event_date: r.date, prog_name: r.prog, position: r.pos })),
});

beforeEach(() => {
  // Two OQR-ranked men from the seed: Vilaca 86042 (#1) and a lower-ranked one.
  installWtFetch([
    { match: "/athletes/86042/results", response: results([
      { event: 1, prog: "Elite Men", pos: 1, date: "2026-05-30" },
      { event: 2, prog: "Elite Men", pos: 2, date: "2026-06-20" },
      { event: 3, prog: "Elite Men", pos: 5, date: "2025-10-15" },
    ]) },
    { match: "/athletes/49390/results", response: results([
      { event: 1, prog: "Elite Men", pos: 4, date: "2026-05-30" }, // Vilaca ahead
      { event: 2, prog: "Elite Men", pos: 1, date: "2026-06-20" }, // 49390 ahead
      { event: 9, prog: "Elite Men", pos: 3, date: "2025-09-01" }, // not common
    ]) },
    { match: "/athletes/86042", response: profile(86042, "POR", { race_starts: 40, race_wins: 8, race_podiums: 20, finish_percentage: 95 }) },
    { match: "/athletes/49390", response: profile(49390, "FRA", { race_starts: 50, race_wins: 3, race_podiums: 25, finish_percentage: 92 }) },
  ]);
});

afterEach(() => vi.unstubAllGlobals());

describe("buildVersus", () => {
  it("compares two ranked athletes and computes the OQR leader", async () => {
    const v = (await buildVersus(86042, 49390))!;
    expect(v).not.toBeNull();
    expect(v.a.athleteId).toBe(86042);
    expect(v.a.oqrRank).toBe(1); // Vilaca leads the seed OQR
    expect(v.oqrLeader).toBe("a");
    expect(v.sameNoc).toBe(false);
    expect(v.sameGender).toBe(true);
  });

  it("builds the head-to-head record from common races only", async () => {
    const v = (await buildVersus(86042, 49390))!;
    // events 1 and 2 are common (event 3 and 9 are not shared) → 2 meetings
    expect(v.h2h.total).toBe(2);
    expect(v.h2h.aWins).toBe(1); // event 1: Vilaca 1 beats 4
    expect(v.h2h.bWins).toBe(1); // event 2: 49390 1 beats 2
    expect(v.h2h.meetings[0].date).toBe("2026-06-20"); // most recent first
  });

  it("returns null for a self-comparison", async () => {
    expect(await buildVersus(86042, 86042)).toBeNull();
  });
});
