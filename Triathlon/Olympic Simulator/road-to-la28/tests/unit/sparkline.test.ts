import { describe, it, expect } from "vitest";
import { sparkGeometry } from "@/lib/trajectory";

describe("sparkGeometry", () => {
  it("inverts the rank axis: the best rank sits at the top", () => {
    const g = sparkGeometry([20, 5], 100, 40, 4); // slipped… no: climbed 20→5
    const [first, last] = g.points;
    // rank 5 is the best (min) → smallest y (top); rank 20 → largest y (bottom).
    expect(last.rank).toBe(5);
    expect(last.y).toBeLessThan(first.y);
    expect(g.bestRank).toBe(5);
    expect(g.worstRank).toBe(20);
  });

  it("reports climb vs slip via `improved` (first − last)", () => {
    expect(sparkGeometry([20, 5]).improved).toBe(15); // climbed 15 places
    expect(sparkGeometry([5, 20]).improved).toBe(-15); // slipped
    expect(sparkGeometry([8, 8]).improved).toBe(0); // held
  });

  it("spaces points evenly across the width", () => {
    const g = sparkGeometry([10, 8, 6], 100, 40, 0);
    expect(g.points.map((p) => p.x)).toEqual([0, 50, 100]);
  });

  it("draws a flat mid-line for an all-equal series (no divide-by-zero)", () => {
    const g = sparkGeometry([3, 3, 3], 120, 40);
    expect(g.points.every((p) => p.y === 20)).toBe(true);
  });

  it("centers a single point", () => {
    const g = sparkGeometry([1], 132, 34);
    expect(g.points).toHaveLength(1);
    expect(g.points[0].x).toBe(66);
    expect(g.improved).toBe(0);
  });
});
