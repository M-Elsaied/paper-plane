import { describe, it, expect } from "vitest";
import { normalizeRanking, type RawRanking } from "@/lib/wt-api/rankings";

function raw(partial: Partial<RawRanking["rankings"][number]>): RawRanking {
  return {
    ranking_id: 11,
    ranking_name: "Elite Men",
    ranking_cat_name: "Olympic",
    published: "2026-06-28",
    total: 1,
    rankings: [
      {
        athlete_id: 1,
        athlete_full_name: "Test Athlete",
        athlete_noc: "GBR",
        athlete_gender: "male",
        rank: 1,
        total: 100,
        scores_current_period: [],
        scores_previous_period: [],
        ...partial,
      },
    ],
  } as RawRanking;
}

describe("normalizeRanking", () => {
  it("coerces change 'NEW' to undefined but keeps numeric change", () => {
    expect(normalizeRanking(raw({ change: "NEW" }), "male").athletes[0].change).toBeUndefined();
    expect(normalizeRanking(raw({ change: 0 }), "male").athletes[0].change).toBe(0);
    expect(normalizeRanking(raw({ change: -3 }), "male").athletes[0].change).toBe(-3);
  });

  it("drops null padding and non-positive score values", () => {
    const a = normalizeRanking(
      raw({ scores_current_period: [1000, null, 0, -5, 925], scores_previous_period: [null, null] }),
      "male",
    ).athletes[0];
    expect(a.scores.map((s) => s.points).sort((x, y) => y - x)).toEqual([1000, 925]);
  });

  it("maps current period → 1 and previous period → 2", () => {
    const a = normalizeRanking(raw({ scores_current_period: [500], scores_previous_period: [400] }), "male").athletes[0];
    expect(a.scores.find((s) => s.points === 500)!.period).toBe(1);
    expect(a.scores.find((s) => s.points === 400)!.period).toBe(2);
  });

  it("falls back to the gender argument when athlete_gender is absent", () => {
    const r = raw({});
    delete (r.rankings[0] as { athlete_gender?: string }).athlete_gender;
    expect(normalizeRanking(r, "female").athletes[0].gender).toBe("female");
  });
});
