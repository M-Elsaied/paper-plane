/**
 * Web Push (VAPID) — subscription storage + sending. Healthy-by-design: the cron
 * only sends when something REAL happened to an athlete the user follows.
 */
import "server-only";
import webpush from "web-push";
import { eq } from "drizzle-orm";
import { getDb } from "@/db/client";
import { pushSubscriptions } from "@/db/schema";

let configured = false;
export function configureWebPush(): boolean {
  if (configured) return true;
  const pub = process.env.VAPID_PUBLIC_KEY;
  const priv = process.env.VAPID_PRIVATE_KEY;
  const subject = process.env.VAPID_SUBJECT || "mailto:hello@example.com";
  if (!pub || !priv) return false;
  webpush.setVapidDetails(subject, pub, priv);
  configured = true;
  return true;
}

export interface BrowserSubscription {
  endpoint: string;
  keys: { p256dh: string; auth: string };
}

export interface PushPayload {
  title: string;
  body: string;
  url?: string;
  tag?: string;
}

export async function saveSubscription(
  accountId: string | null,
  sub: BrowserSubscription,
  follows: number[],
): Promise<boolean> {
  const db = getDb();
  if (!db) return false;
  await db
    .insert(pushSubscriptions)
    .values({
      accountId: accountId ?? undefined,
      endpoint: sub.endpoint,
      p256dh: sub.keys.p256dh,
      auth: sub.keys.auth,
      follows,
    })
    .onConflictDoUpdate({
      target: pushSubscriptions.endpoint,
      set: { accountId: accountId ?? undefined, follows, p256dh: sub.keys.p256dh, auth: sub.keys.auth },
    });
  return true;
}

export async function deleteSubscription(endpoint: string) {
  const db = getDb();
  if (!db) return;
  await db.delete(pushSubscriptions).where(eq(pushSubscriptions.endpoint, endpoint));
}

type SubRow = typeof pushSubscriptions.$inferSelect;

/** Send a payload; on 404/410 (expired) delete the subscription. Returns sent. */
export async function sendToSubscription(row: SubRow, payload: PushPayload): Promise<boolean> {
  if (!configureWebPush()) return false;
  try {
    await webpush.sendNotification(
      { endpoint: row.endpoint, keys: { p256dh: row.p256dh, auth: row.auth } },
      JSON.stringify(payload),
    );
    return true;
  } catch (err) {
    const status = (err as { statusCode?: number }).statusCode;
    if (status === 404 || status === 410) await deleteSubscription(row.endpoint);
    return false;
  }
}

export async function allSubscriptions(): Promise<SubRow[]> {
  const db = getDb();
  if (!db) return [];
  return db.select().from(pushSubscriptions);
}

export async function subscriptionsForEndpoint(endpoint: string): Promise<SubRow[]> {
  const db = getDb();
  if (!db) return [];
  return db.select().from(pushSubscriptions).where(eq(pushSubscriptions.endpoint, endpoint));
}
