import { describe, it, expect } from "vitest";
import { testDb } from "../helpers/setup";
import { rankingSnapshots, rankingEntries } from "@/db/schema";
import { getRankTrajectory } from "@/lib/data";
import { buildCockpit } from "@/lib/cockpit";
import seedMen from "@/data/qual-state-men.json";

async function snapshot(hash: string, publishedAt: string, fetchedAt: string) {
  const [s] = await testDb()
    .insert(rankingSnapshots)
    .values({ rankingId: 11, rankingType: "oqr_men", contentHash: hash, publishedAt, fetchedAt: new Date(fetchedAt) })
    .returning({ id: rankingSnapshots.id });
  return s.id;
}

describe("rank trajectory", () => {
  it("reads an athlete's rank across stored OQR snapshots, oldest → newest", async () => {
    const s1 = await snapshot("h1", "2026-05-01", "2026-05-01T00:00:00Z");
    const s2 = await snapshot("h2", "2026-06-01", "2026-06-01T00:00:00Z");
    const s3 = await snapshot("h3", "2026-07-01", "2026-07-01T00:00:00Z");
    await testDb()
      .insert(rankingEntries)
      .values([
        { snapshotId: s1, athleteId: 999, rank: 40, lastRank: 50 },
        { snapshotId: s2, athleteId: 999, rank: 25, lastRank: 40 },
        { snapshotId: s3, athleteId: 999, rank: 12, lastRank: 25 },
      ]);

    const traj = await getRankTrajectory("male", 999);
    // Oldest snapshot's lastRank (50) is prepended, then the three snapshot ranks.
    expect(traj.map((p) => p.rank)).toEqual([50, 40, 25, 12]);
    expect(traj[0].label).toBe("prev");
    expect(traj[traj.length - 1].date).toBe("2026-07-01");
  });

  it("ignores snapshots the athlete is absent from", async () => {
    const s1 = await snapshot("h1", "2026-05-01", "2026-05-01T00:00:00Z");
    const s2 = await snapshot("h2", "2026-06-01", "2026-06-01T00:00:00Z");
    await testDb()
      .insert(rankingEntries)
      .values([
        { snapshotId: s1, athleteId: 111, rank: 3, lastRank: 4 }, // a different athlete
        { snapshotId: s2, athleteId: 999, rank: 8, lastRank: 9 },
      ]);
    const traj = await getRankTrajectory("male", 999);
    // Only the s2 rank (8) exists for 999; its lastRank (9) is prepended.
    expect(traj.map((p) => p.rank)).toEqual([9, 8]);
  });

  it("falls back to the athlete's own last→now official ranks when there is no history", async () => {
    // No snapshots inserted → getQualState uses seed JSON; the cockpit synthesizes
    // a two-point trend from the athlete's stored lastRank.
    // Any seeded athlete with a previous rank works; derive the expectation from
    // the committed seed so a reseed (ranks move) doesn't break the test.
    const seeded = seedMen.athletes.find((a) => a.athleteId === 56027)!; // Diego Moya
    expect(seeded.lastRank).toBeTypeOf("number");
    const m = (await buildCockpit(56027))!;
    expect(m).not.toBeNull();
    expect(m.trajectory).toHaveLength(2);
    expect(m.trajectory[0].rank).toBe(seeded.lastRank);
    expect(m.trajectory[0].label).toBe("last ranking");
    expect(m.trajectory[1].label).toBe("now");
    expect(m.trajectory[1].rank).toBe(m.rank);
  });
});
