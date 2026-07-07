import { describe, it, expect } from "vitest";
import { athleteStatus } from "@/lib/engine/status";
import type { QualLine } from "@/lib/engine/types";

function line(overrides: Partial<QualLine>): QualLine {
  return {
    gender: "male",
    qualified: [],
    cutRank: 21,
    cutPoints: 600,
    bubble: [],
    perNocUsage: {},
    skipped: [],
    ...overrides,
  };
}

describe("athleteStatus", () => {
  it("marks a qualified athlete IN", () => {
    const l = line({ qualified: [{ athleteId: 1, fullName: "A", noc: "GBR", rank: 5, total: 900 }] });
    expect(athleteStatus(l, 1).code).toBe("in");
    expect(athleteStatus(l, 1).label).toBe("QUALIFYING");
  });

  it("marks a cap-blocked athlete BLOCKED", () => {
    const l = line({ skipped: [{ athleteId: 2, fullName: "B", noc: "FRA", rank: 10, reason: "noc_cap", detail: "FRA cap reached (3)" }] });
    const s = athleteStatus(l, 2);
    expect(s.code).toBe("blocked");
    expect(s.detail).toContain("FRA");
  });

  it("marks an ineligible athlete INELIGIBLE", () => {
    const l = line({ skipped: [{ athleteId: 3, fullName: "C", noc: "USA", rank: 200, reason: "ineligible", detail: "outside top 160" }] });
    expect(athleteStatus(l, 3).code).toBe("ineligible");
  });

  it("marks anyone else CHASING", () => {
    expect(athleteStatus(line({}), 999).code).toBe("chasing");
  });
});
