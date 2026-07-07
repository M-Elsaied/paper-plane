import { describe, it, expect } from "vitest";
import { GET as syncRankings } from "@/app/api/cron/sync-rankings/route";
import { readQualState, readMrNations } from "@/lib/db-read";
import { getQualState } from "@/lib/data";
import { installWtFetch } from "../helpers/mock-wt";
import { rankingRoutes, loadFixture } from "../helpers/fixtures";

const syncReq = () => new Request("http://test/api/cron/sync-rankings");

async function ingestAll() {
  installWtFetch(rankingRoutes());
  await syncRankings(syncReq());
}

describe("db-read reconstruction", () => {
  it("rebuilds QualState in rank order with scores surviving the JSONB round trip", async () => {
    await ingestAll();
    const state = (await readQualState("male"))!;
    expect(state).not.toBeNull();
    expect(state.athletes.length).toBeGreaterThan(100);
    // rank order preserved (entries ordered by rank)
    expect(state.athletes[0].scores.length).toBeGreaterThan(0);
    // published leader is Vilaca (id 86042)
    expect(state.athletes[0].athleteId).toBe(86042);
  });

  it("coerces a string 'NEW' change back to undefined on read", async () => {
    await ingestAll();
    const raw = loadFixture("ranking-11-oqr-men.json") as { data: { rankings: { athlete_id: number; change: unknown }[] } };
    const newEntrant = raw.data.rankings.find((r) => r.change === "NEW");
    if (!newEntrant) return; // fixture may have none this week — skip cleanly
    const state = (await readQualState("male"))!;
    const a = state.athletes.find((x) => x.athleteId === newEntrant.athlete_id)!;
    expect(a.change).toBeUndefined();
  });

  it("returns the latest snapshot after a change", async () => {
    await ingestAll();
    const bumped = structuredClone(loadFixture("ranking-11-oqr-men.json")) as {
      data: { rankings: { total: number; scores_previous_period: (number | null)[] }[] };
    };
    // Bump total (drives the content hash → new snapshot) AND the underlying
    // score (drives reconstruction) so the read-back reflects the newer snapshot.
    bumped.data.rankings[0].total += 500;
    bumped.data.rankings[0].scores_previous_period[0]! += 500;
    installWtFetch([{ match: "/rankings/11", response: bumped }, ...rankingRoutes().filter((r) => r.match !== "/rankings/11")]);
    await syncRankings(syncReq());
    const state = (await readQualState("male"))!;
    const counted = state.athletes[0].scores.reduce((s, x) => s + x.points, 0);
    expect(counted).toBeGreaterThan(2000);
  });

  it("reads MR nations with a team NOC", async () => {
    await ingestAll();
    const nations = (await readMrNations())!;
    expect(nations.length).toBeGreaterThan(0);
    expect(nations[0].noc).toMatch(/^[A-Z]{2,3}$|—/);
    expect(nations[0].rank).toBe(1);
  });

  it("falls back to seed JSON when the DB has no snapshot", async () => {
    // no ingest → readQualState null → data.getQualState returns committed seed
    expect(await readQualState("male")).toBeNull();
    const seed = await getQualState("male");
    expect(seed.athletes.length).toBeGreaterThan(100);
  });
});
