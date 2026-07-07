/**
 * GOLDEN TESTS — the product's credibility claim.
 * Recompute every athlete's best-12/max-7 total from the same score arrays the
 * app ingests, and assert it equals World Triathlon's PUBLISHED total. Catches
 * any regression in normalization (null padding, period mapping, "NEW") or
 * selection (caps) against reality. Refresh fixtures with: npm run fixtures:record -- --force
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect } from "vitest";
import { normalizeRanking, type RawRanking } from "@/lib/wt-api/rankings";
import { selectCountingScores } from "@/lib/engine/best-scores";
import { rankAthletes } from "@/lib/engine/qualification";
import type { Gender } from "@/config/pathways";

function load(file: string) {
  const raw = readFileSync(join(process.cwd(), "tests/fixtures/wt", file), "utf8");
  return JSON.parse(raw) as { data: RawRanking };
}

const GENDERS: { file: string; gender: Gender }[] = [
  { file: "ranking-11-oqr-men.json", gender: "male" },
  { file: "ranking-12-oqr-women.json", gender: "female" },
];

// A few athletes pinned by name for readable failures (the "golden ten" core).
const PINNED = ["Vasco Vilaca", "Dorian Coninx", "Ricardo Batista", "Miguel Hidalgo"];

describe("golden: engine totals == published WT totals", () => {
  for (const g of GENDERS) {
    const raw = load(g.file).data;
    const state = normalizeRanking(raw, g.gender);
    const publishedById = new Map(raw.rankings.map((r) => [r.athlete_id, r]));

    it(`${g.gender}: every athlete's counted total matches published (±0.005)`, () => {
      const offenders: { name: string; computed: number; published: number }[] = [];
      state.athletes.forEach((a) => {
        const computed = selectCountingScores(a.scores).total;
        const published = publishedById.get(a.athleteId)!.total;
        if (Math.abs(computed - published) > 0.005) {
          offenders.push({ name: a.fullName, computed, published });
        }
      });
      expect(offenders).toEqual([]);
      expect(state.athletes.length).toBeGreaterThan(100); // sanity: real payload
    });

    it(`${g.gender}: no counted set exceeds 12 total or 7 per period`, () => {
      for (const a of state.athletes) {
        const c = selectCountingScores(a.scores);
        expect(c.counted.length).toBeLessThanOrEqual(12);
        expect(c.perPeriodCount[1]).toBeLessThanOrEqual(7);
        expect(c.perPeriodCount[2]).toBeLessThanOrEqual(7);
      }
    });

    it(`${g.gender}: our ranking's top 20 == published top 20 (set)`, () => {
      const ours = rankAthletes(state.athletes).slice(0, 20).map((a) => a.athleteId).sort();
      const published = [...raw.rankings]
        .sort((a, b) => a.rank - b.rank)
        .slice(0, 20)
        .map((r) => r.athlete_id)
        .sort();
      expect(ours).toEqual(published);
    });
  }

  it("pinned athletes reproduce their published total exactly", () => {
    const raw = load("ranking-11-oqr-men.json").data;
    const state = normalizeRanking(raw, "male");
    for (const name of PINNED) {
      const a = state.athletes.find((x) => x.fullName === name);
      expect(a, `athlete ${name} present`).toBeDefined();
      const published = raw.rankings.find((r) => r.athlete_id === a!.athleteId)!.total;
      expect(selectCountingScores(a!.scores).total).toBeCloseTo(published, 2);
    }
  });
});
