/**
 * Event-driven notifications. After a NEW official ranking is ingested, diff it
 * against the previous one and push each subscriber a message ONLY about the
 * athletes they follow — line crossings and real rank moves. Healthy by design:
 * no message unless something genuinely happened. Deduped per snapshot.
 */
import "server-only";
import { desc, eq, and } from "drizzle-orm";
import type { Db } from "@/db/client";
import { rankingSnapshots, rankingEntries, qualificationStates, athletes } from "@/db/schema";
import type { Gender } from "@/config/pathways";
import type { QualLine } from "@/lib/engine/types";
import { allSubscriptions, sendToSubscription, type PushPayload } from "@/lib/push";
import { pushSubscriptions } from "@/db/schema";

interface AthleteEvent {
  athleteId: number;
  name: string;
  kind: "crossed_in" | "dropped_out" | "moved_up" | "moved_down";
  rank: number;
  delta: number;
}

async function twoLatest(db: Db, rankingType: string) {
  return db
    .select()
    .from(rankingSnapshots)
    .where(eq(rankingSnapshots.rankingType, rankingType))
    .orderBy(desc(rankingSnapshots.fetchedAt))
    .limit(2);
}

async function lineFor(db: Db, snapshotId: number, gender: Gender): Promise<QualLine | null> {
  const [row] = await db
    .select()
    .from(qualificationStates)
    .where(and(eq(qualificationStates.snapshotId, snapshotId), eq(qualificationStates.gender, gender)))
    .limit(1);
  return (row?.line as QualLine) ?? null;
}

/** Returns number of pushes sent for this gender. */
export async function notifyOnRankingUpdate(db: Db, gender: Gender): Promise<number> {
  const type = gender === "male" ? "oqr_men" : "oqr_women";
  const snaps = await twoLatest(db, type);
  if (snaps.length < 2) return 0; // need a baseline to diff
  const [newSnap, prevSnap] = snaps;

  const [newLine, prevLine] = await Promise.all([
    lineFor(db, newSnap.id, gender),
    lineFor(db, prevSnap.id, gender),
  ]);
  const newQual = new Set(newLine?.qualified.map((q) => q.athleteId) ?? []);
  const prevQual = new Set(prevLine?.qualified.map((q) => q.athleteId) ?? []);

  // New snapshot entries + names.
  const rows = await db
    .select({
      athleteId: rankingEntries.athleteId,
      rank: rankingEntries.rank,
      change: rankingEntries.change,
      name: athletes.fullName,
    })
    .from(rankingEntries)
    .leftJoin(athletes, eq(rankingEntries.athleteId, athletes.athleteId))
    .where(eq(rankingEntries.snapshotId, newSnap.id));

  const events = new Map<number, AthleteEvent>();
  for (const r of rows) {
    if (r.athleteId == null) continue;
    const change = Number(r.change);
    const delta = Number.isFinite(change) ? change : 0;
    const name = r.name ?? "Your athlete";
    if (newQual.has(r.athleteId) && !prevQual.has(r.athleteId)) {
      events.set(r.athleteId, { athleteId: r.athleteId, name, kind: "crossed_in", rank: r.rank, delta });
    } else if (!newQual.has(r.athleteId) && prevQual.has(r.athleteId)) {
      events.set(r.athleteId, { athleteId: r.athleteId, name, kind: "dropped_out", rank: r.rank, delta });
    } else if (delta >= 2) {
      events.set(r.athleteId, { athleteId: r.athleteId, name, kind: "moved_up", rank: r.rank, delta });
    } else if (delta <= -2) {
      events.set(r.athleteId, { athleteId: r.athleteId, name, kind: "moved_down", rank: r.rank, delta });
    }
  }
  if (!events.size) return 0;

  const subs = await allSubscriptions();
  let sent = 0;

  for (const sub of subs) {
    const already = gender === "male" ? sub.lastSnapshotMen : sub.lastSnapshotWomen;
    if (already === newSnap.id) continue; // deduped
    const follows = (sub.follows as number[] | null) ?? [];
    const hit = follows.map((id) => events.get(id)).filter(Boolean) as AthleteEvent[];

    if (hit.length) {
      const ok = await sendToSubscription(sub, buildPayload(hit));
      if (ok) sent++;
    }
    // Mark this snapshot processed for this sub regardless (avoids re-sends).
    await db
      .update(pushSubscriptions)
      .set(gender === "male" ? { lastSnapshotMen: newSnap.id } : { lastSnapshotWomen: newSnap.id })
      .where(eq(pushSubscriptions.id, sub.id));
  }
  return sent;
}

function buildPayload(events: AthleteEvent[]): PushPayload {
  const cross = events.find((e) => e.kind === "crossed_in");
  const drop = events.find((e) => e.kind === "dropped_out");
  const first = events[0];

  if (events.length === 1) {
    const e = events[0];
    const body =
      e.kind === "crossed_in"
        ? `${e.name} crossed the qualification line — now #${e.rank}! 🥇`
        : e.kind === "dropped_out"
          ? `${e.name} slipped out of the qualifying places, now #${e.rank}.`
          : e.kind === "moved_up"
            ? `${e.name} climbed ${e.delta} to #${e.rank} on the new ranking.`
            : `${e.name} dropped ${Math.abs(e.delta)} to #${e.rank}.`;
    return { title: "Road to LA28 · new ranking", body, url: `/athlete/${e.athleteId}`, tag: "ranking" };
  }

  const headline = cross
    ? `${cross.name} crossed the line — `
    : drop
      ? `${drop.name} slipped out — `
      : "";
  return {
    title: "Road to LA28 · new ranking",
    body: `${headline}${events.length} of your athletes moved. Tap to see what changed.`,
    url: "/pulse",
    tag: "ranking",
  };
}
