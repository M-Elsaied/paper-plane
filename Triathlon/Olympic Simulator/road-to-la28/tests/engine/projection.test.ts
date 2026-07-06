import { describe, it, expect } from "vitest";
import { seedFinishOrder, pointsForOrder, projectRanking, type FieldAthlete } from "@/lib/engine/projection";
import type { AthleteScores } from "@/lib/engine/types";

let id = 0;
function ath(noc: string, total: number, change = 0): AthleteScores & { total: number; change: number } {
  return {
    athleteId: ++id,
    fullName: `${noc}-${total}`,
    noc,
    gender: "male",
    scores: total > 0 ? [{ points: total, period: 1 }] : [],
    change,
    total,
  } as AthleteScores & { total: number; change: number };
}

function field(a: ReturnType<typeof ath>[]): FieldAthlete[] {
  return a.map((x, i) => ({ ...x, currentTotal: x.total, currentRank: i + 1, change: x.change }));
}

describe("seedFinishOrder", () => {
  it("orders stronger athletes ahead", () => {
    const f = field([ath("A", 900), ath("B", 300), ath("C", 600)]);
    const order = seedFinishOrder(f).map((s) => s.athleteId);
    // strongest (A=900) first, then C=600, then B=300
    expect(order[0]).toBe(f[0].athleteId);
    expect(order[order.length - 1]).toBe(f[1].athleteId);
  });

  it("gives rising form a boost between similar athletes", () => {
    const hot = ath("A", 500, 10);
    const cold = ath("B", 500, -10);
    const order = seedFinishOrder(field([cold, hot])).map((s) => s.athleteId);
    expect(order[0]).toBe(hot.athleteId);
  });

  it("orders by strength even when change is a non-number like 'NEW'", () => {
    // Regression: the WT API sends change:"NEW" for new entrants, which used to
    // poison maxAbsChange -> NaN seeds -> the sort no-op'd (start-list order).
    const strong = ath("A", 1800, 0);
    const weakNew = ath("B", 50, 0);
    const f = field([weakNew, strong]);
    (f[0] as { change: unknown }).change = "NEW";
    const order = seedFinishOrder(f).map((s) => s.athleteId);
    expect(order[0]).toBe(strong.athleteId);
  });
});

describe("pointsForOrder", () => {
  it("assigns descending points by finish position", () => {
    const map = pointsForOrder([10, 20, 30], "wtcs");
    expect(map.get(10)!.position).toBe(1);
    expect(map.get(10)!.expectedPoints).toBeGreaterThan(map.get(20)!.expectedPoints);
    expect(map.get(20)!.expectedPoints).toBeGreaterThan(map.get(30)!.expectedPoints);
  });
});

describe("projectRanking", () => {
  it("moves a bubble athlete up and can cross the line with a projected win", () => {
    // 24 athletes, our target sits well outside the 21-place line.
    const pool = Array.from({ length: 23 }, (_, i) => ath(`N${i}`, 1000 - i * 10));
    const target = ath("POR", 120);
    const all: AthleteScores[] = [...pool, target];
    const f = field([target]); // only the target is racing (simplest field)
    f[0].currentTotal = 120;
    f[0].currentRank = 24;

    // Target projected to WIN → gets ~1000 pts.
    const proj = projectRanking(all, f, [target.athleteId], "wtcs", 2);
    const row = proj.rows.find((r) => r.athleteId === target.athleteId)!;
    expect(row.expectedPoints).toBeGreaterThan(900);
    expect(row.projectedRank).toBeLessThan(row.currentRank!);
    expect(row.pointsDelta).toBeGreaterThan(0);
  });

  it("produces a projected line at the individual slot count", () => {
    const pool = Array.from({ length: 40 }, (_, i) => ath(`Z${i}`, 900 - i));
    const f = field([pool[30]]);
    const proj = projectRanking(pool, f, [pool[30].athleteId], "wtcs", 2);
    expect(proj.projectedLine.qualified.length).toBe(21);
  });
});
