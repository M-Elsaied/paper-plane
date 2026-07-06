import { describe, it, expect } from "vitest";
import { applyWhatIf } from "@/lib/engine/what-if";
import type { QualState, AthleteScores } from "@/lib/engine/types";
import { DEFAULT_ASSUMPTIONS } from "@/config/pathways";

let id = 0;
function ath(noc: string, total: number): AthleteScores {
  return {
    athleteId: ++id,
    fullName: `${noc}-${total}`,
    noc,
    gender: "male",
    scores: [{ points: total, period: 1 }],
  };
}

const noHost = { ...DEFAULT_ASSUMPTIONS, host: { noc: "USA", perGender: 0 } };

describe("applyWhatIf", () => {
  it("moves a bubble athlete across the line with a strong result", () => {
    // 21 slots. Make 25 athletes; our target sits at rank 25 (well outside).
    const pool = Array.from({ length: 24 }, (_, i) => ath(`N${i}`, 1000 - i * 10));
    const target = ath("POR", 100); // clearly last
    const state: QualState = {
      gender: "male",
      publishedAt: "2026-07-01",
      rankingId: 11,
      athletes: [...pool, target],
    };

    const before = applyWhatIf(state, { athleteId: target.athleteId, tier: "wtcs", position: 25, period: 2 }, noHost);
    expect(before.before.qualified).toBe(false);

    // A win at a WTCS event (1000 pts) should vault them inside.
    const res = applyWhatIf(state, { athleteId: target.athleteId, tier: "wtcs", position: 1, period: 2 }, noHost);
    expect(res.hypotheticalPoints).toBeGreaterThan(900);
    expect(res.after.qualified).toBe(true);
    expect(res.crossesLine).toBe(true);
    expect(res.rankDelta).toBeGreaterThan(0);
  });

  it("reports no line crossing when the athlete is already safely inside", () => {
    const pool = Array.from({ length: 30 }, (_, i) => ath(`M${i}`, 1000 - i * 10));
    const leader = pool[0];
    const state: QualState = { gender: "male", publishedAt: "2026-07-01", rankingId: 11, athletes: pool };
    const res = applyWhatIf(state, { athleteId: leader.athleteId, tier: "wtcs", position: 1, period: 2 }, noHost);
    expect(res.before.qualified).toBe(true);
    expect(res.after.qualified).toBe(true);
    expect(res.crossesLine).toBe(false);
  });
});
