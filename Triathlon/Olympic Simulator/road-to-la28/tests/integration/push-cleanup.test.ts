import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("web-push", () => ({ default: { setVapidDetails: vi.fn(), sendNotification: vi.fn() } }));
import webpush from "web-push";

import { sendToSubscription } from "@/lib/push";
import { getDb } from "@/db/client";
import { pushSubscriptions } from "@/db/schema";

const sendMock = vi.mocked(webpush.sendNotification);

async function seedSub() {
  const db = getDb()!;
  const [row] = await db
    .insert(pushSubscriptions)
    .values({ endpoint: "https://push/expired", p256dh: "x", auth: "y" })
    .returning();
  return row;
}

beforeEach(() => {
  process.env.VAPID_PUBLIC_KEY = "pub";
  process.env.VAPID_PRIVATE_KEY = "priv";
  sendMock.mockReset();
});

const payload = { title: "t", body: "b" };

describe("push send + expired-subscription cleanup", () => {
  it("deletes a subscription on a 410 Gone", async () => {
    const row = await seedSub();
    sendMock.mockRejectedValueOnce(Object.assign(new Error("gone"), { statusCode: 410 }));
    const sent = await sendToSubscription(row, payload);
    expect(sent).toBe(false);
    const remaining = await getDb()!.select().from(pushSubscriptions);
    expect(remaining.length).toBe(0);
  });

  it("keeps the subscription on a transient 500", async () => {
    const row = await seedSub();
    sendMock.mockRejectedValueOnce(Object.assign(new Error("boom"), { statusCode: 500 }));
    expect(await sendToSubscription(row, payload)).toBe(false);
    expect((await getDb()!.select().from(pushSubscriptions)).length).toBe(1);
  });

  it("returns true on success", async () => {
    const row = await seedSub();
    sendMock.mockResolvedValueOnce({} as never);
    expect(await sendToSubscription(row, payload)).toBe(true);
  });

  it("returns false (no send) when VAPID keys are unset", async () => {
    delete process.env.VAPID_PUBLIC_KEY;
    delete process.env.VAPID_PRIVATE_KEY;
    // configureWebPush caches; force a fresh module state by re-importing
    vi.resetModules();
    const { sendToSubscription: fresh } = await import("@/lib/push");
    const row = await seedSub();
    expect(await fresh(row, payload)).toBe(false);
    expect(sendMock).not.toHaveBeenCalled();
  });
});
