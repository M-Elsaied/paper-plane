import { describe, it, expect, afterEach, vi } from "vitest";
import { buildRaceCompanion, getEvent } from "@/lib/race";
import { installWtFetch } from "../helpers/mock-wt";
import { hamburgRoutes } from "../helpers/fixtures";

// Runs in the unit project (no DB) → getQualState falls back to committed seed JSON.
afterEach(() => vi.unstubAllGlobals());

describe("buildRaceCompanion", () => {
  it("uses the official start list and counts unranked entrants", async () => {
    installWtFetch(hamburgRoutes());
    const m = (await buildRaceCompanion(195148, "male"))!;
    expect(m).not.toBeNull();
    expect(m.officialStartList).toBe(true);
    expect(m.field.length).toBeGreaterThan(10);
    expect(m.unrankedEntrants).toBeGreaterThanOrEqual(0);
    // seeded finish order is deterministic and starts with a real contender
    expect(m.initialOrder[0]).toBe(m.field.map((f) => f.athleteId).find((id) => m.initialOrder[0] === id));
    expect(m.tier).toBe("wtcs");
  });

  it("falls back to an expected field when the start list can't be fetched", async () => {
    installWtFetch([]); // any /events fetch throws → caught → fallback
    const m = (await buildRaceCompanion(195148, "male"))!;
    expect(m.officialStartList).toBe(false);
    expect(m.field.length).toBe(30); // top-30 expected field
  });

  it("returns null for an unknown event", async () => {
    expect(getEvent(424242)).toBeNull();
    expect(await buildRaceCompanion(424242, "male")).toBeNull();
  });
});
