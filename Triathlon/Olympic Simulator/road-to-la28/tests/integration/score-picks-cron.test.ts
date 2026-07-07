import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("web-push", () => ({
  default: { setVapidDetails: vi.fn(), sendNotification: vi.fn().mockResolvedValue({}) },
}));
import webpush from "web-push";

import { GET } from "@/app/api/cron/score-picks/route";
import { getDb } from "@/db/client";
import { picks, pushSubscriptions } from "@/db/schema";
import { PICKEM } from "@/config/pickem";
import { installWtFetch } from "../helpers/mock-wt";
import { yokohamaRoutes } from "../helpers/fixtures";
import { eq } from "drizzle-orm";

const req = () => new Request("http://test/api/cron/score-picks");
const sendMock = vi.mocked(webpush.sendNotification);

beforeEach(() => {
  process.env.VAPID_PUBLIC_KEY = "pub";
  process.env.VAPID_PRIVATE_KEY = "priv";
  sendMock.mockClear();
});

// Yokohama Elite Men actual podium (fixture): Hauser 80795, Hidalgo 105480, Willian 55495.
describe("score-picks cron", () => {
  it("scores a perfect podium and a wrong-order podium against the official result", async () => {
    const db = getDb()!;
    await db.insert(picks).values([
      { raceId: 195145, gender: "male", ownerKey: "dev:d1", podium: [80795, 105480, 55495] }, // perfect
      { raceId: 195145, gender: "male", ownerKey: "acc-1", podium: [105480, 55495, 80795] }, // all-podium, no exact
    ]);
    await db.insert(pushSubscriptions).values({ endpoint: "https://p/a", p256dh: "x", auth: "y", accountId: "acc-1" });

    installWtFetch(yokohamaRoutes());
    const body = await (await GET(req())).json();
    expect(body.status).toBe("ok");
    expect(body.summary["195145:male"]).toBe("scored:2");

    const rows = await db.select().from(picks).where(eq(picks.raceId, 195145));
    const perfect = rows.find((r) => r.ownerKey === "dev:d1")!;
    expect(perfect.perfect).toBe(true);
    expect(perfect.score).toBe(PICKEM.exact[1] + PICKEM.exact[2] + PICKEM.exact[3] + PICKEM.perfectBonus); // 25
    const wrong = rows.find((r) => r.ownerKey === "acc-1")!;
    expect(wrong.perfect).toBe(false);
    expect(wrong.score).toBe(3 * PICKEM.onPodiumWrongSpot); // 6

    // only the account owner gets a push, not the dev owner
    expect(sendMock).toHaveBeenCalledTimes(1);
    expect(JSON.parse(sendMock.mock.calls[0][1] as string).body).toContain("Results are in");
  });

  it("does not re-score an already-scored pick", async () => {
    const db = getDb()!;
    await db.insert(picks).values({ raceId: 195145, gender: "male", ownerKey: "dev:d1", podium: [1, 2, 3], score: 7, scoredAt: new Date() });
    installWtFetch(yokohamaRoutes());
    await GET(req());
    const [row] = await db.select().from(picks).where(eq(picks.raceId, 195145));
    expect(row.score).toBe(7); // untouched
  });

  it("leaves picks pending when results aren't in yet", async () => {
    const db = getDb()!;
    await db.insert(picks).values({ raceId: 999999, gender: "male", ownerKey: "dev:d1", podium: [1, 2, 3] });
    installWtFetch([
      { match: "/events/999999/programs/555/results", response: { code: 200, status: "ok", data: { results: [] } } },
      { match: "/events/999999/programs", response: { code: 200, status: "ok", data: [{ prog_id: 555, prog_name: "Elite Men" }] } },
    ]);
    const body = await (await GET(req())).json();
    expect(body.summary["999999:male"]).toBe("pending");
    const [row] = await db.select().from(picks).where(eq(picks.raceId, 999999));
    expect(row.score).toBeNull();
  });
});
