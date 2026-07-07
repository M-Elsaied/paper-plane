import { describe, it, expect } from "vitest";
import { eq } from "drizzle-orm";
import { POST, GET } from "@/app/api/picks/route";
import { signToken } from "@/lib/auth";
import { getDb } from "@/db/client";
import { picks } from "@/db/schema";
import { cookieJar } from "../helpers/setup";

const post = (body: unknown) =>
  POST(new Request("http://test/api/picks", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }));
const get = (qs: string) => GET(new Request(`http://test/api/picks?${qs}`));

const DEVICE = "device-abcdef12";

describe("picks API — validation & fairness", () => {
  it("rejects malformed podiums", async () => {
    expect((await post({ raceId: 1, gender: "male", podium: [1, 1, 2], deviceId: DEVICE })).status).toBe(400); // dup
    expect((await post({ raceId: 1, gender: "male", podium: [1, 2], deviceId: DEVICE })).status).toBe(400); // too few
    expect((await post({ raceId: 1, gender: "male", podium: [1, 2, 3] })).status).toBe(400); // no owner (no device/session)
    expect((await post({ gender: "male", podium: [1, 2, 3], deviceId: DEVICE })).status).toBe(400); // no race
  });

  it("accepts a valid pick keyed by device", async () => {
    const res = await post({ raceId: 10, gender: "male", podium: [1, 2, 3], deviceId: DEVICE });
    expect(res.status).toBe(200);
    const rows = await getDb()!.select().from(picks);
    expect(rows.length).toBe(1);
    expect(rows[0].ownerKey).toBe(`dev:${DEVICE}`);
  });

  it("prefers the session account over the device id as owner", async () => {
    cookieJar().set("rtla28_session", signToken("acc-XYZ"));
    await post({ raceId: 11, gender: "male", podium: [4, 5, 6], deviceId: DEVICE });
    const [row] = await getDb()!.select().from(picks).where(eq(picks.raceId, 11));
    expect(row.ownerKey).toBe("acc-XYZ");
  });

  it("re-submitting before results resets any score (fairness)", async () => {
    await post({ raceId: 12, gender: "male", podium: [1, 2, 3], deviceId: DEVICE });
    const db = getDb()!;
    await db.update(picks).set({ score: 25, perfect: true, scoredAt: new Date() }).where(eq(picks.raceId, 12));
    await post({ raceId: 12, gender: "male", podium: [3, 2, 1], deviceId: DEVICE }); // change pick
    const [row] = await db.select().from(picks).where(eq(picks.raceId, 12));
    expect(row.score).toBeNull();
    expect(row.perfect).toBeNull();
    expect(row.podium).toEqual([3, 2, 1]);
  });

  it("aggregates the crowd forecast (win = slot 1 only, podium = any slot)", async () => {
    await post({ raceId: 20, gender: "male", podium: [100, 200, 300], deviceId: "device-aaaaaaa1" });
    await post({ raceId: 20, gender: "male", podium: [100, 300, 400], deviceId: "device-bbbbbbb2" });
    await post({ raceId: 20, gender: "male", podium: [200, 100, 300], deviceId: "device-ccccccc3" });

    const body = await (await get("raceId=20&gender=male&deviceId=device-zzzzzzz9")).json();
    expect(body.crowd.total).toBe(3);
    const win100 = body.crowd.athletes.find((a: { athleteId: number }) => a.athleteId === 100);
    expect(win100.win).toBe(2); // picked 1st twice
    expect(win100.podium).toBe(3); // on all three podiums
    expect(body.crowd.athletes[0].athleteId).toBe(100); // sorted: most wins first
  });
});
