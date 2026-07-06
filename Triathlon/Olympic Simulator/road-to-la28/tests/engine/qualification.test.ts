import { describe, it, expect } from "vitest";
import {
  computeQualificationLine,
  computeNocCaps,
  rankAthletes,
} from "@/lib/engine/qualification";
import type { AthleteScores } from "@/lib/engine/types";
import { DEFAULT_ASSUMPTIONS } from "@/config/pathways";

let id = 0;
function ath(noc: string, total: number, gender: "male" | "female" = "male"): AthleteScores {
  return {
    athleteId: ++id,
    fullName: `${noc}-${total}`,
    noc,
    gender,
    scores: [{ points: total, period: 1 }],
  };
}

describe("qualification line", () => {
  it("ranks by counted total descending", () => {
    const ranked = rankAthletes([ath("A", 100), ath("B", 300), ath("C", 200)]);
    expect(ranked.map((r) => r.total)).toEqual([300, 200, 100]);
  });

  it("grants the 3-athlete cap only to nations with depth in the top 30", () => {
    // FRA has 3 athletes near the top; everyone else is a distinct NOC.
    const pool = [
      ath("FRA", 1000),
      ath("FRA", 990),
      ath("FRA", 980),
      ...Array.from({ length: 30 }, (_, i) => ath(`N${i}`, 900 - i)),
    ];
    const ranked = rankAthletes(pool);
    const caps = computeNocCaps(ranked);
    expect(caps["FRA"]).toBe(3);
    expect(caps["N0"]).toBe(2);
  });

  it("caps a deep nation at 3 and blocks its 4th athlete", () => {
    // 4 GBR near the top => depth rule grants cap 3; the 4th is blocked. (A
    // nation can only ever be cap-2 if it has <3 contenders, so the meaningful
    // block is the cap-3 case.)
    const pool = [
      ath("GBR", 1000),
      ath("GBR", 995),
      ath("GBR", 990),
      ath("GBR", 985),
      ...Array.from({ length: 40 }, (_, i) => ath(`X${i}`, 900 - i)),
    ];
    const line = computeQualificationLine(pool, {
      ...DEFAULT_ASSUMPTIONS,
      host: { noc: "USA", perGender: 0 }, // isolate the cap behaviour
    });
    const gbrQualified = line.qualified.filter((q) => q.noc === "GBR");
    expect(gbrQualified.length).toBe(3);
    const gbrSkip = line.skipped.find((s) => s.noc === "GBR" && s.reason === "noc_cap");
    expect(gbrSkip).toBeDefined();
  });

  it("fills exactly the individual slot count", () => {
    const pool = Array.from({ length: 60 }, (_, i) => ath(`Z${i}`, 900 - i));
    const line = computeQualificationLine(pool, {
      ...DEFAULT_ASSUMPTIONS,
      host: { noc: "USA", perGender: 0 },
    });
    expect(line.qualified.length).toBe(DEFAULT_ASSUMPTIONS.individualSlots);
    expect(line.cutRank).toBe(DEFAULT_ASSUMPTIONS.individualSlots);
  });

  it("counts host pathway places against the host nation's cap", () => {
    // USA has 3 strong athletes and depth => cap 3. Host consumes 2, so only 1
    // USA athlete can additionally take an individual slot.
    const pool = [
      ath("USA", 1000),
      ath("USA", 995),
      ath("USA", 990),
      ...Array.from({ length: 30 }, (_, i) => ath(`Y${i}`, 900 - i)),
    ];
    const line = computeQualificationLine(pool, {
      ...DEFAULT_ASSUMPTIONS,
      host: { noc: "USA", perGender: 2 },
    });
    const usaIndividual = line.qualified.filter((q) => q.noc === "USA");
    expect(usaIndividual.length).toBe(1);
  });
});
