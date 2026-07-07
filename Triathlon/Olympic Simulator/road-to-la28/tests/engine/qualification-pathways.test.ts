import { describe, it, expect } from "vitest";
import { computeQualificationLine } from "@/lib/engine/qualification";
import { DEFAULT_ASSUMPTIONS } from "@/config/pathways";
import type { AthleteScores } from "@/lib/engine/types";

let id = 0;
function ath(noc: string, total: number, extra: Partial<AthleteScores> = {}): AthleteScores {
  return { athleteId: ++id, fullName: `${noc}-${total}`, noc, gender: "male", scores: [{ points: total, period: 1 }], ...extra };
}
const noHost = { ...DEFAULT_ASSUMPTIONS, host: { noc: "USA", perGender: 0 } };

describe("pathway consumption + eligibility", () => {
  it("an MR World Champ nation consumes an individual slot (dead-conditional regression)", () => {
    // NOR has exactly 2 near the top (cap 2). The MR-champ pathway consumes 1,
    // so only 1 NOR athlete can also take an individual slot; the other is
    // blocked with reason pathway_consumed. (Before the fix this consumed 0.)
    const pool = [
      ath("NOR", 1000),
      ath("NOR", 990),
      ...Array.from({ length: 30 }, (_, i) => ath(`N${i}`, 900 - i)),
    ];
    const line = computeQualificationLine(pool, { ...noHost, mrWorldChamps2026: "NOR" });
    expect(line.qualified.filter((q) => q.noc === "NOR").length).toBe(1);
    expect(line.skipped.some((s) => s.noc === "NOR" && s.reason === "pathway_consumed")).toBe(true);
  });

  it("caps the bubble at 8 and cuts at the individual slot count", () => {
    const pool = Array.from({ length: 45 }, (_, i) => ath(`Z${i}`, 900 - i));
    const line = computeQualificationLine(pool, noHost);
    expect(line.qualified.length).toBe(21);
    expect(line.cutRank).toBe(21);
    expect(line.bubble.length).toBeLessThanOrEqual(8);
  });

  it("skips an explicitly ineligible athlete and gives the slot to the next", () => {
    const pool = [
      ath("AAA", 1000),
      ath("BBB", 990, { eligible: false }), // would be #2 but ineligible
      ...Array.from({ length: 30 }, (_, i) => ath(`C${i}`, 900 - i)),
    ];
    const line = computeQualificationLine(pool, noHost);
    expect(line.qualified.some((q) => q.noc === "BBB")).toBe(false);
    expect(line.skipped.some((s) => s.noc === "BBB" && s.reason === "ineligible")).toBe(true);
    expect(line.qualified.length).toBe(21);
  });
});
