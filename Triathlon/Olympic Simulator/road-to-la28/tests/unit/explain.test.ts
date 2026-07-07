import { describe, it, expect } from "vitest";
import { explainPosition } from "@/lib/explain";
import type { CockpitModel } from "@/lib/cockpit";

function model(o: Partial<CockpitModel>): CockpitModel {
  return {
    athleteId: 1,
    fullName: "Marcus Cook",
    noc: "GBR",
    gender: "male",
    rank: 10,
    total: 700,
    qualified: false,
    gapToLine: 50,
    cutRank: 21,
    cutPoints: 600,
    counted: [],
    marginalPoints: 120,
    periodCount: { 1: 3, 2: 2 },
    periodFull: { 1: false, 2: false },
    daysToDeadline: 682,
    mr: { noc: "GBR", rank: 4, total: 700, insideTop16: true, gapToTop16: 0, worldChampsSlot: null },
    status: { code: "chasing", label: "CHASING", tone: "electric", detail: "" },
    nocUsage: { cap: 2, used: 1 },
    nocAhead: 0,
    chasers: [],
    publishedAt: "2026-06-28",
    ...o,
  } as CockpitModel;
}

describe("explainPosition", () => {
  it("leads with the qualifying story and the points buffer", () => {
    const lines = explainPosition(model({ status: { code: "in", label: "QUALIFYING", tone: "good", detail: "" }, gapToLine: -120 }));
    expect(lines[0].text).toContain("inside the 21 individual places");
    expect(lines[0].text).toContain("120");
    expect(lines[0].tone).toBe("good");
  });

  it("explains a cap block naming the nation and teammates ahead", () => {
    const lines = explainPosition(model({ status: { code: "blocked", label: "BLOCKED", tone: "warn", detail: "" }, nocUsage: { cap: 3, used: 3 }, nocAhead: 3 }));
    expect(lines[0].text).toContain("GBR");
    expect(lines[0].text).toContain("out-point a teammate");
  });

  it("names the points gap when chasing", () => {
    const lines = explainPosition(model({ gapToLine: 50, cutRank: 21 }));
    expect(lines[0].text).toContain("50 points below the cut");
    expect(lines[0].text).toContain("rank 21");
  });

  it("flags a full period", () => {
    const lines = explainPosition(model({ periodFull: { 1: true, 2: false } }));
    expect(lines.some((l) => l.text.includes("Period 1 is full"))).toBe(true);
  });

  it("adds the Mixed Relay pathway line when inside the top 16", () => {
    const lines = explainPosition(model({}));
    expect(lines.some((l) => l.text.includes("Mixed Relay") && l.text.includes("top 16"))).toBe(true);
  });
});
