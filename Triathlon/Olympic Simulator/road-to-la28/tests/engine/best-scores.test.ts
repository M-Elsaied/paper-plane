import { describe, it, expect } from "vitest";
import { selectCountingScores } from "@/lib/engine/best-scores";
import type { Score } from "@/lib/engine/types";

const s = (points: number, period: 1 | 2): Score => ({ points, period });

describe("selectCountingScores", () => {
  it("takes the best 12 across periods", () => {
    const scores = Array.from({ length: 20 }, (_, i) => s(1000 - i * 10, i % 2 === 0 ? 1 : 2));
    const r = selectCountingScores(scores);
    expect(r.counted.length).toBe(12);
    // marginal is the lowest counted score
    expect(r.marginal!.points).toBe(r.counted[r.counted.length - 1].points);
  });

  it("enforces max 7 from a single period", () => {
    // 10 strong scores all in period 1, 3 weak in period 2.
    const scores = [
      ...Array.from({ length: 10 }, (_, i) => s(900 - i, 1)),
      ...Array.from({ length: 3 }, (_, i) => s(100 - i, 2)),
    ];
    const r = selectCountingScores(scores);
    expect(r.perPeriodCount[1]).toBe(7); // capped
    expect(r.periodFull[1]).toBe(true);
    // 12 total = 7 from P1 + best available from P2 (only 3 exist)
    expect(r.counted.length).toBe(10);
  });

  it("handles fewer than 12 scores", () => {
    const r = selectCountingScores([s(500, 1), s(400, 2)]);
    expect(r.total).toBe(900);
    expect(r.counted.length).toBe(2);
  });

  it("sums the counted total correctly", () => {
    const r = selectCountingScores([s(1000, 1), s(925, 1)]);
    expect(r.total).toBe(1925);
  });
});
