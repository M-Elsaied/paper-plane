import { describe, it, expect } from "vitest";
import { analyzeRoad, computeQualifiedNocs, type RoadContext, type WorldRankAthlete } from "@/lib/engine/road";
import { rankAthletes, computeQualificationLine } from "@/lib/engine/qualification";
import { DEFAULT_ASSUMPTIONS } from "@/config/pathways";
import type { AthleteScores } from "@/lib/engine/types";
import type { MrNationEntry } from "@/lib/engine/mixed-relay";

let id = 0;
function ath(noc: string, total: number): AthleteScores {
  return { athleteId: ++id, fullName: `${noc} Athlete-${total}`, noc, gender: "male", scores: [{ points: total, period: 1 }] };
}

function ctx(
  pool: AthleteScores[],
  mr: MrNationEntry[],
  subjectId: number,
  fallback?: RoadContext["subjectFallback"],
  worldRanking?: WorldRankAthlete[],
): RoadContext {
  const ranked = rankAthletes(pool);
  const line = computeQualificationLine(pool, { ...DEFAULT_ASSUMPTIONS, host: { noc: "USA", perGender: 0 } });
  return { ranked, line, mrNations: mr, worldRanking, assumptions: { ...DEFAULT_ASSUMPTIONS, host: { noc: "USA", perGender: 0 } }, subjectId, subjectFallback: fallback };
}

describe("analyzeRoad", () => {
  it("gives a top-ranked athlete the individual route as primary, on track", () => {
    const pool = [ath("POR", 1900), ...Array.from({ length: 40 }, (_, i) => ath(`N${i}`, 1500 - i * 20))];
    const road = analyzeRoad(ctx(pool, [], pool[0].athleteId));
    expect(road.primary?.key).toBe("individual");
    expect(road.primary?.status).toBe("on_track");
    expect(road.subject.continent).toBe("Europe"); // POR
  });

  it("surfaces teammate cap rivals on the individual route", () => {
    // 4 FRA athletes near the top → 4th is cap-blocked; teammates are the rivals.
    const pool = [ath("FRA", 1900), ath("FRA", 1850), ath("FRA", 1800), ath("FRA", 1750), ...Array.from({ length: 40 }, (_, i) => ath(`N${i}`, 1500 - i * 20))];
    const fra4 = pool[3];
    const road = analyzeRoad(ctx(pool, [], fra4.athleteId));
    const indiv = road.routes.find((r) => r.key === "individual")!;
    expect(indiv.status).toBe("locked_out");
    expect(indiv.competitors.some((c) => c.noc === "FRA" && c.ahead)).toBe(true);
  });

  it("gives an emerging-nation athlete New Flag routes with continental rivals", () => {
    // EGY athlete outside the line; other African not-yet-qualified nations present.
    const pool = [
      ...Array.from({ length: 30 }, (_, i) => ath(`EUR${i}`, 1600 - i * 10)),
      ath("EGY", 300),
      ath("RSA", 320),
      ath("MAR", 280),
    ];
    const egy = pool.find((a) => a.noc === "EGY")!;
    const road = analyzeRoad(ctx(pool, [], egy.athleteId));
    const keys = road.routes.map((r) => r.key);
    expect(keys).toContain("newflag_continental");
    expect(keys).toContain("newflag_ranking");
    const nf = road.routes.find((r) => r.key === "newflag_continental")!;
    // RSA (Africa, higher total) is a continental rival ahead
    expect(nf.competitors.some((c) => c.noc === "RSA")).toBe(true);
    expect(road.subject.continent).toBe("Africa");
  });

  it("treats an unranked emerging athlete honestly (individual locked, New Flag present)", () => {
    const pool = [
      ...Array.from({ length: 25 }, (_, i) => ath(`EUR${i}`, 1600 - i * 10)),
      ath("KEN", 200),
    ];
    // subject not in the pool at all (truly unranked)
    const road = analyzeRoad(ctx(pool, [], 999999, { name: "Test Runner", noc: "EGY", gender: "male", worldRank: null }));
    expect(road.subject.oqrRank).toBeNull();
    const indiv = road.routes.find((r) => r.key === "individual")!;
    expect(indiv.status).toBe("locked_out");
    expect(road.routes.some((r) => r.key === "newflag_ranking")).toBe(true);
  });

  it("excludes New Flag for a nation that already holds a place", () => {
    const pool = [ath("GBR", 1900), ...Array.from({ length: 30 }, (_, i) => ath(`N${i}`, 1500 - i * 20))];
    const gbr = pool[0];
    const road = analyzeRoad(ctx(pool, [], gbr.athleteId));
    expect(road.routes.some((r) => r.key.startsWith("newflag"))).toBe(false);
  });

  it("draws New Flag rivals from the full World Ranking, not just the OQR pool", () => {
    // OQR pool: only Europeans + one EGY subject. The African rivals (MAR, RSA)
    // exist ONLY in the World Ranking — they must still surface as New Flag rivals.
    const pool = [...Array.from({ length: 25 }, (_, i) => ath(`EUR${i}`, 1600 - i * 10)), ath("EGY", 250)];
    const egy = pool.find((a) => a.noc === "EGY")!;
    const world: WorldRankAthlete[] = [
      { athleteId: egy.athleteId, fullName: "EGY Subject", noc: "EGY", rank: 300 },
      { athleteId: 9001, fullName: "Morocco Runner", noc: "MAR", rank: 29 },
      { athleteId: 9002, fullName: "South Africa Runner", noc: "RSA", rank: 142 },
      { athleteId: 9003, fullName: "Kenya Runner", noc: "KEN", rank: 260 },
    ];
    const road = analyzeRoad(ctx(pool, [], egy.athleteId, undefined, world));
    expect(road.subject.worldRank).toBe(300);
    const nf = road.routes.find((r) => r.key === "newflag_ranking")!;
    const nocs = nf.competitors.map((c) => c.noc);
    expect(nocs).toContain("MAR"); // only in the World Ranking, not the OQR pool
    expect(nocs).toContain("RSA");
    // MAR (#29) is ahead of the subject (#300); KEN (#260) is ahead too
    expect(nf.competitors.find((c) => c.noc === "MAR")!.ahead).toBe(true);
    // 3 African rivals rank ahead → in_contention needs <=2, so this is a stretch
    expect(nf.status).toBe("stretch");
  });

  it("marks New Flag on_track when the subject is their continent's best not-yet-qualified athlete", () => {
    const pool = [...Array.from({ length: 25 }, (_, i) => ath(`EUR${i}`, 1600 - i * 10)), ath("EGY", 250)];
    const egy = pool.find((a) => a.noc === "EGY")!;
    const world: WorldRankAthlete[] = [
      { athleteId: egy.athleteId, fullName: "EGY Subject", noc: "EGY", rank: 40 },
      { athleteId: 9002, fullName: "South Africa Runner", noc: "RSA", rank: 142 },
    ];
    const road = analyzeRoad(ctx(pool, [], egy.athleteId, undefined, world));
    const nf = road.routes.find((r) => r.key === "newflag_ranking")!;
    expect(nf.status).toBe("on_track"); // #40 beats the only African rival (#142)
  });

  it("computes qualified NOCs from line + relay top 8", () => {
    const pool = [ath("GBR", 1900), ath("ESP", 1850)];
    const mr: MrNationEntry[] = [{ noc: "FRA", rank: 1, total: 900 }, { noc: "GER", rank: 2, total: 800 }];
    const set = computeQualifiedNocs(ctx(pool, mr, pool[0].athleteId));
    expect(set.has("GBR")).toBe(true); // in line
    expect(set.has("FRA")).toBe(true); // relay top 8
  });
});
