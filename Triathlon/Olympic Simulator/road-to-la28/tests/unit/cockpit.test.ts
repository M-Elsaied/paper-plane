import { describe, it, expect } from "vitest";
import { buildCockpit, buildRanking } from "@/lib/cockpit";

// Unit project (no DB) → reads committed seed JSON.
describe("cockpit view-model", () => {
  it("builds the #1 athlete as qualifying with a positive buffer", async () => {
    const m = (await buildCockpit(86042))!; // Vasco Vilaca
    expect(m).not.toBeNull();
    expect(m.rank).toBe(1);
    expect(m.noc).toBe("POR");
    expect(m.status.code).toBe("in");
    expect(m.qualified).toBe(true);
    expect(m.gapToLine).toBeLessThan(0); // inside the line
    expect(m.counted.length).toBeGreaterThan(0);
  });

  it("returns null for an unknown athlete", async () => {
    expect(await buildCockpit(424242)).toBeNull();
  });

  it("builds a ranking whose cut sits at the individual slot count", async () => {
    const { line, rows } = await buildRanking("male");
    expect(line.cutRank).toBe(21);
    expect(rows[0].rank).toBe(1);
    expect(rows.find((r) => r.athleteId === 86042)!.qualified).toBe(true);
  });
});
