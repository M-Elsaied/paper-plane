import { describe, it, expect } from "vitest";
import { assembleWarRoom } from "@/lib/war-room";
import type { RankingRow } from "@/lib/cockpit";
import type { QualLine, Gender } from "@/lib/engine/types";
import type { MrNationEntry } from "@/lib/engine/mixed-relay";

// Minimal builders — assembleWarRoom only reads rows + line.perNocUsage + line.skipped.
const row = (athleteId: number, rank: number, noc: string, total: number, qualified: boolean): RankingRow => ({
  athleteId,
  rank,
  fullName: `A${athleteId}`,
  noc,
  total,
  qualified,
});

const line = (
  gender: Gender,
  perNocUsage: QualLine["perNocUsage"],
  skipped: QualLine["skipped"] = [],
): QualLine => ({
  gender,
  qualified: [],
  cutRank: null,
  cutPoints: null,
  bubble: [],
  perNocUsage,
  skipped,
});

describe("assembleWarRoom", () => {
  // FRA: 3 men, cap 2 → 2 qualify, #6 locked out. POR/GBR: 1 each. USA: host, 2 pathway places, no ranked men.
  const menRows = [
    row(1, 1, "FRA", 1900, true),
    row(2, 2, "FRA", 1850, true),
    row(3, 3, "POR", 1800, true),
    row(4, 4, "GBR", 1700, true),
    row(6, 6, "FRA", 1500, false),
  ];
  const menLine = line(
    "male",
    { FRA: { cap: 2, used: 2 }, POR: { cap: 2, used: 1 }, GBR: { cap: 2, used: 1 }, USA: { cap: 3, used: 2 } },
    [{ athleteId: 6, fullName: "A6", noc: "FRA", rank: 6, reason: "noc_cap", detail: "FRA cap reached (2)" }],
  );
  const womenRows = [row(11, 1, "GBR", 1950, true), row(12, 2, "FRA", 1900, true)];
  const womenLine = line("female", {
    GBR: { cap: 2, used: 1 },
    FRA: { cap: 2, used: 1 },
    USA: { cap: 3, used: 2 },
  });
  const mrNations: MrNationEntry[] = [
    { noc: "FRA", rank: 1, total: 4000 },
    { noc: "GBR", rank: 2, total: 3900 },
    { noc: "USA", rank: 5, total: 3500 },
  ];

  const model = assembleWarRoom({
    men: { rows: menRows, line: menLine },
    women: { rows: womenRows, line: womenLine },
    mrNations,
    publishedAt: "2026-05-18",
  });

  it("groups both genders by NOC and sums secured places", () => {
    const fra = model.nocs.find((n) => n.noc === "FRA")!;
    expect(fra.name).toBe("France");
    expect(fra.men.secured).toBe(2);
    expect(fra.women.secured).toBe(1);
    expect(fra.securedTotal).toBe(3);
    expect(fra.men.contenders).toHaveLength(3); // 3 ranked French men
  });

  it("flags athletes locked out by the nation cap", () => {
    const fra = model.nocs.find((n) => n.noc === "FRA")!;
    expect(fra.men.blockedCount).toBe(1);
    const lockedOut = fra.men.contenders.find((c) => c.athleteId === 6)!;
    expect(lockedOut.blockedByCap).toBe(true);
    expect(lockedOut.qualified).toBe(false);
  });

  it("keeps a pathway-only host nation with no ranked athletes", () => {
    const usa = model.nocs.find((n) => n.noc === "USA")!;
    expect(usa).toBeTruthy();
    expect(usa.contenderTotal).toBe(0);
    expect(usa.securedTotal).toBe(4); // 2 men + 2 women via the host pathway
    expect(usa.men.pathwaySecured).toBe(2);
  });

  it("computes open slots as cap minus secured", () => {
    const gbr = model.nocs.find((n) => n.noc === "GBR")!;
    expect(gbr.men.open).toBe(1); // cap 2, 1 secured
    expect(gbr.women.open).toBe(1);
  });

  it("folds in Mixed Relay standing and sorts powerhouses first", () => {
    const fra = model.nocs.find((n) => n.noc === "FRA")!;
    expect(fra.mr.rank).toBe(1);
    expect(fra.mr.insideRelayCut).toBe(true);
    // FRA has the most secured places (3) → sorts ahead of USA (4)… USA has more.
    expect(model.nocs[0].noc).toBe("USA"); // 4 secured
    expect(model.nocs[1].noc).toBe("FRA"); // 3 secured
    expect(model.totals.mrNationsInside).toBe(3);
  });
});
