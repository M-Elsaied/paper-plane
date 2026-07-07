import { describe, it, expect, afterEach } from "vitest";
import { eq } from "drizzle-orm";
import { GET } from "@/app/api/cron/sync-rankings/route";
import { getDb } from "@/db/client";
import { rankingSnapshots, rankingEntries, athletes, qualificationStates } from "@/db/schema";
import { installWtFetch } from "../helpers/mock-wt";
import { rankingRoutes, loadFixture } from "../helpers/fixtures";
import { setTestDb } from "../helpers/setup";

const req = (headers?: Record<string, string>) =>
  new Request("http://test/api/cron/sync-rankings", { headers });

afterEach(() => {
  delete process.env.CRON_SECRET;
});

describe("sync-rankings cron", () => {
  it("ingests 5 rankings: snapshots, scored entries, athletes, qualification states", async () => {
    installWtFetch(rankingRoutes());
    const res = await GET(req());
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.summary.oqr_men).toMatch(/^updated:\d+/);
    expect(body.summary.mr_olympic).toMatch(/^updated:\d+/);

    const db = getDb()!;
    const snaps = await db.select().from(rankingSnapshots);
    expect(snaps.length).toBe(5);

    const menSnap = snaps.find((s) => s.rankingType === "oqr_men")!;
    const entries = await db.select().from(rankingEntries).where(eq(rankingEntries.snapshotId, menSnap.id));
    expect(entries.length).toBeGreaterThan(100);
    // scores survive as JSONB and are non-empty for the leader
    const leader = entries.find((e) => e.rank === 1)!;
    expect(Array.isArray(leader.scores)).toBe(true);
    expect((leader.scores as unknown[]).length).toBeGreaterThan(0);

    const ath = await db.select().from(athletes);
    expect(ath.length).toBeGreaterThan(100);

    const qs = await db.select().from(qualificationStates);
    expect(qs.map((q) => q.gender).sort()).toEqual(["female", "male"]);
    expect((qs[0].line as { qualified: unknown[] }).qualified.length).toBeGreaterThan(0);
  });

  it("is idempotent — a second identical run is all 'unchanged'", async () => {
    installWtFetch(rankingRoutes());
    await GET(req());
    const res2 = await GET(req());
    const body2 = await res2.json();
    expect(Object.values(body2.summary).every((v) => v === "unchanged" || String(v).startsWith("sent"))).toBe(true);
    const snaps = await getDb()!.select().from(rankingSnapshots);
    expect(snaps.length).toBe(5); // no new snapshots
  });

  it("detects a change and writes only the changed ranking", async () => {
    installWtFetch(rankingRoutes());
    await GET(req()); // baseline

    const bumped = structuredClone(loadFixture("ranking-11-oqr-men.json")) as { data: { rankings: { total: number }[] } };
    bumped.data.rankings[0].total += 123; // change the leader's total → new hash
    installWtFetch([{ match: "/rankings/11", response: bumped }, ...rankingRoutes().filter((r) => r.match !== "/rankings/11")]);

    const res = await GET(req());
    const body = await res.json();
    expect(body.summary.oqr_men).toMatch(/^updated/);
    expect(body.summary.oqr_women).toBe("unchanged");

    const menSnaps = (await getDb()!.select().from(rankingSnapshots)).filter((s) => s.rankingType === "oqr_men");
    expect(menSnaps.length).toBe(2);
  });

  it("refuses a renamed ranking (guards against WT renumbering)", async () => {
    const wrong = structuredClone(loadFixture("ranking-11-oqr-men.json")) as { data: { ranking_name: string } };
    wrong.data.ranking_name = "Totally Different";
    installWtFetch([{ match: "/rankings/11", response: wrong }, ...rankingRoutes().filter((r) => r.match !== "/rankings/11")]);

    const res = await GET(req());
    const body = await res.json();
    expect(body.summary.oqr_men).toMatch(/^name-mismatch/);
    const menSnaps = (await getDb()!.select().from(rankingSnapshots)).filter((s) => s.rankingType === "oqr_men");
    expect(menSnaps.length).toBe(0); // nothing written for the mismatched ranking
  });

  it("401s without the cron bearer when CRON_SECRET is set", async () => {
    process.env.CRON_SECRET = "s3cret";
    installWtFetch(rankingRoutes());
    const res = await GET(req());
    expect(res.status).toBe(401);
    const ok = await GET(req({ authorization: "Bearer s3cret" }));
    expect(ok.status).toBe(200);
  });

  it("no-ops without a database", async () => {
    setTestDb(null);
    const res = await GET(req());
    const body = await res.json();
    expect(body.status).toBe("noop");
  });
});
