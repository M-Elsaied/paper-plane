import { describe, it, expect, afterEach, vi } from "vitest";
import { buildRaceCompanion, getEvent } from "@/lib/race";
import { installWtFetch } from "../helpers/mock-wt";
import { hamburgRoutes, hamburgEventRoute, noCalendarRoute } from "../helpers/fixtures";

// Runs in the unit project (no DB) → getQualState falls back to committed seed JSON.
// Hamburg (195148) has raced, so it's no longer in the seed calendar: the builder
// must resolve it live via /events/{id} (fixture). The calendar list call itself
// is unmocked → throws → seed-calendar fallback.
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
    installWtFetch(hamburgEventRoute()); // event resolves; /programs → 404 → caught → fallback
    const m = (await buildRaceCompanion(195148, "male"))!;
    expect(m).not.toBeNull();
    expect(m.officialStartList).toBe(false);
    expect(m.field.length).toBe(30); // top-30 expected field
  });

  it("returns null for an unknown event", async () => {
    // Calendar 404s fast; the live /events/424242 lookup 404s too → null, no retries.
    installWtFetch([
      ...noCalendarRoute(),
      { match: "/events/424242", response: () => ({ body: { code: 404, status: "error" }, status: 404 }) },
    ]);
    expect(await getEvent(424242)).toBeNull();
    expect(await buildRaceCompanion(424242, "male")).toBeNull();
  });
});
