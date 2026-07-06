import { describe, it, expect } from "vitest";
import { scorePick } from "@/lib/engine/pickem";
import { PICKEM } from "@/config/pickem";

describe("scorePick", () => {
  const actual = [10, 20, 30]; // 1st=10, 2nd=20, 3rd=30

  it("scores a perfect podium with the bonus", () => {
    const r = scorePick([10, 20, 30], actual);
    expect(r.perfect).toBe(true);
    expect(r.score).toBe(10 + 6 + 4 + PICKEM.perfectBonus);
    expect(r.correctCount).toBe(3);
  });

  it("awards partial credit for right athlete, wrong slot", () => {
    // predicted winner (10) finishes 1st exact; 30 predicted 2nd but is on podium
    const r = scorePick([10, 30, 99], actual);
    expect(r.slots[0].outcome).toBe("exact");
    expect(r.slots[1].outcome).toBe("podium");
    expect(r.slots[2].outcome).toBe("miss");
    expect(r.score).toBe(10 + PICKEM.onPodiumWrongSpot);
    expect(r.perfect).toBe(false);
  });

  it("scores zero for an all-miss podium", () => {
    const r = scorePick([1, 2, 3], actual);
    expect(r.score).toBe(0);
    expect(r.correctCount).toBe(0);
  });

  it("does not award the perfect bonus for 2/3 exact", () => {
    const r = scorePick([10, 20, 99], actual);
    expect(r.perfect).toBe(false);
    expect(r.score).toBe(10 + 6);
  });
});
