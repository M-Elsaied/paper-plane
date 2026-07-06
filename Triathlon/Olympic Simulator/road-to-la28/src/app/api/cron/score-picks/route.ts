/**
 * Score-picks cron. For every race with unscored picks, fetch the official
 * podium; if results are in, auto-score all picks and notify the owners whose
 * predictions have landed. Safe to run often — no results yet = no-op.
 */
import { NextResponse } from "next/server";
import { eq, inArray } from "drizzle-orm";
import { getDb } from "@/db/client";
import { pushSubscriptions } from "@/db/schema";
import { authorizeCron, startRun, finishRun } from "@/lib/ingest/sync-run";
import { racesNeedingScoring, scoreRace } from "@/lib/picks";
import { fetchRacePodiums } from "@/lib/wt-api/results";
import { sendToSubscription } from "@/lib/push";

export const maxDuration = 300;
export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  if (!authorizeCron(req)) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const db = getDb();
  if (!db) return NextResponse.json({ status: "noop", note: "no database" });

  const runId = await startRun("score-picks");
  const summary: Record<string, string> = {};
  try {
    const races = await racesNeedingScoring();
    // Group genders by race to fetch podiums once per event.
    const byEvent = new Map<number, Set<string>>();
    for (const r of races) {
      if (!byEvent.has(r.raceId)) byEvent.set(r.raceId, new Set());
      byEvent.get(r.raceId)!.add(r.gender);
    }

    for (const [eventId, genders] of byEvent) {
      const podiums = await fetchRacePodiums(eventId);
      for (const gender of genders) {
        const actual = gender === "male" ? podiums.male : podiums.female;
        if (actual.length < 3) {
          summary[`${eventId}:${gender}`] = "pending";
          continue;
        }
        const { scored, owners } = await scoreRace(eventId, gender as "male" | "female", actual);
        summary[`${eventId}:${gender}`] = `scored:${scored}`;
        await notifyOwners(owners, eventId);
      }
    }
    await finishRun(runId, "ok", summary);
    return NextResponse.json({ status: "ok", summary });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await finishRun(runId, "error", summary, message);
    return NextResponse.json({ status: "error", error: message, summary }, { status: 500 });
  }
}

/** Push "your pick is in" to account-linked owners with a subscription. */
async function notifyOwners(owners: string[], eventId: number) {
  const db = getDb();
  if (!db || !owners.length) return;
  const accountIds = owners.filter((o) => !o.startsWith("dev:"));
  if (!accountIds.length) return;
  const subs = await db
    .select()
    .from(pushSubscriptions)
    .where(inArray(pushSubscriptions.accountId, accountIds));
  for (const sub of subs) {
    await sendToSubscription(sub, {
      title: "Road to LA28 · Pick-'Em",
      body: "Results are in — see how your podium call scored. 🏅",
      url: `/race/${eventId}`,
      tag: "pickem",
    });
  }
}
