import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock web-push before importing anything that pulls it in.
vi.mock("web-push", () => ({
  default: { setVapidDetails: vi.fn(), sendNotification: vi.fn().mockResolvedValue({}) },
}));
import webpush from "web-push";

import { GET as sync } from "@/app/api/cron/sync-rankings/route";
import { notifyOnRankingUpdate } from "@/lib/ingest/notify";
import { getDb } from "@/db/client";
import { pushSubscriptions } from "@/db/schema";
import { installWtFetch } from "../helpers/mock-wt";
import { rankingRoutes, loadFixture } from "../helpers/fixtures";

const req = () => new Request("http://test/api/cron/sync-rankings");
const sendMock = vi.mocked(webpush.sendNotification);

beforeEach(() => {
  process.env.VAPID_PUBLIC_KEY = "pub";
  process.env.VAPID_PRIVATE_KEY = "priv";
  process.env.VAPID_SUBJECT = "mailto:t@t.com";
  sendMock.mockClear();
});

type MenFixture = {
  data: {
    rankings: { athlete_id: number; athlete_noc: string; total: number; scores_previous_period: (number | null)[] }[];
  };
};

/** Vault the athlete at `idx` above the qualification line. Give them an
 *  uncontested NOC so no cap/host pre-consumption blocks them at #1. */
function crossingFixture(idx: number) {
  const f = structuredClone(loadFixture("ranking-11-oqr-men.json")) as MenFixture;
  const r = f.data.rankings[idx];
  r.total += 5000;
  r.scores_previous_period[0] = (r.scores_previous_period[0] ?? 0) + 5000;
  r.athlete_noc = "TST";
  return { fixture: f, athleteId: r.athlete_id };
}

async function syncMen(menResponse?: unknown) {
  const routes = rankingRoutes();
  if (menResponse) {
    installWtFetch([{ match: "/rankings/11", response: menResponse }, ...routes.filter((r) => r.match !== "/rankings/11")]);
  } else {
    installWtFetch(routes);
  }
  await sync(req());
}

describe("event-driven notifications", () => {
  it("pushes only the follower of an athlete who crosses the line", async () => {
    await syncMen(); // baseline
    const { fixture, athleteId } = crossingFixture(25); // rank ~26, outside the 21

    const db = getDb()!;
    // Sub A follows the crossing athlete; Sub B follows the stable leader (86042).
    await db.insert(pushSubscriptions).values([
      { endpoint: "https://push/A", p256dh: "a", auth: "a", accountId: "accA", follows: [athleteId] },
      { endpoint: "https://push/B", p256dh: "b", auth: "b", accountId: "accB", follows: [86042] },
    ]);

    await syncMen(fixture); // new snapshot; notify runs inside the route

    expect(sendMock).toHaveBeenCalledTimes(1);
    const payload = JSON.parse(sendMock.mock.calls[0][1] as string);
    expect(payload.body).toContain("crossed the qualification line");
    expect(payload.url).toBe(`/athlete/${athleteId}`);
  });

  it("dedupes — a repeat notify for the same snapshot sends nothing", async () => {
    await syncMen();
    const { fixture, athleteId } = crossingFixture(25);
    await getDb()!.insert(pushSubscriptions).values({ endpoint: "https://push/A", p256dh: "a", auth: "a", follows: [athleteId] });
    await syncMen(fixture); // sends once (marks lastSnapshotMen)
    sendMock.mockClear();

    const n = await notifyOnRankingUpdate(getDb()!, "male");
    expect(n).toBe(0);
    expect(sendMock).not.toHaveBeenCalled();
  });

  it("sends nothing with only one snapshot (no baseline to diff)", async () => {
    await syncMen();
    await getDb()!.insert(pushSubscriptions).values({ endpoint: "https://push/A", p256dh: "a", auth: "a", follows: [86042] });
    const n = await notifyOnRankingUpdate(getDb()!, "male");
    expect(n).toBe(0);
  });
});
