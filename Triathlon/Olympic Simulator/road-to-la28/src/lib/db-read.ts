/**
 * Neon read helpers — reconstruct engine state from the latest stored snapshot.
 * Returns null when there is no database or no snapshot yet, so the data layer
 * can fall back to seed JSON. Every UI read goes through src/lib/data.ts, which
 * calls these first.
 */
import "server-only";
import { desc, eq, and } from "drizzle-orm";
import { getDb } from "@/db/client";
import { rankingSnapshots, rankingEntries, athletes } from "@/db/schema";
import type { Gender } from "@/config/pathways";
import type { QualState, AthleteScores, Score } from "@/lib/engine/types";
import type { MrNationEntry } from "@/lib/engine/mixed-relay";

async function latestSnapshot(rankingType: string) {
  const db = getDb();
  if (!db) return null;
  const [snap] = await db
    .select()
    .from(rankingSnapshots)
    .where(eq(rankingSnapshots.rankingType, rankingType))
    .orderBy(desc(rankingSnapshots.fetchedAt))
    .limit(1);
  return snap ?? null;
}

/** Rebuild an OQR QualState (athletes + scores) from Neon, or null. */
export async function readQualState(gender: Gender): Promise<QualState | null> {
  const db = getDb();
  if (!db) return null;
  const snap = await latestSnapshot(gender === "male" ? "oqr_men" : "oqr_women");
  if (!snap) return null;

  const rows = await db
    .select({
      athleteId: rankingEntries.athleteId,
      rank: rankingEntries.rank,
      lastRank: rankingEntries.lastRank,
      change: rankingEntries.change,
      totalPoints: rankingEntries.totalPoints,
      scores: rankingEntries.scores,
      fullName: athletes.fullName,
      noc: athletes.noc,
      gender: athletes.gender,
      yearOfBirth: athletes.yearOfBirth,
      headshotUrl: athletes.headshotUrl,
      flagUrl: athletes.flagUrl,
    })
    .from(rankingEntries)
    .innerJoin(athletes, eq(rankingEntries.athleteId, athletes.athleteId))
    .where(eq(rankingEntries.snapshotId, snap.id))
    .orderBy(rankingEntries.rank);

  if (!rows.length) return null;

  const list: AthleteScores[] = rows.map((r) => ({
    athleteId: r.athleteId!,
    fullName: r.fullName,
    noc: r.noc ?? "",
    gender: (r.gender as Gender) ?? gender,
    yearOfBirth: r.yearOfBirth ?? undefined,
    profileImage: r.headshotUrl ?? undefined,
    flag: r.flagUrl ?? undefined,
    publishedRank: r.rank,
    lastRank: r.lastRank ?? undefined,
    change: r.change != null ? Number(r.change) : undefined,
    scores: (r.scores as Score[] | null) ?? [],
  }));

  return { gender, publishedAt: snap.publishedAt ?? "", rankingId: snap.rankingId, athletes: list };
}

/** Mixed Relay nations from the latest MR snapshot, or null. */
export async function readMrNations(): Promise<MrNationEntry[] | null> {
  const db = getDb();
  if (!db) return null;
  const snap = await latestSnapshot("mr_olympic");
  if (!snap) return null;
  const rows = await db
    .select({ noc: rankingEntries.nationNoc, rank: rankingEntries.rank, total: rankingEntries.totalPoints })
    .from(rankingEntries)
    .where(eq(rankingEntries.snapshotId, snap.id))
    .orderBy(rankingEntries.rank);
  if (!rows.length) return null;
  return rows.map((r) => ({ noc: r.noc ?? "—", rank: r.rank, total: r.total ?? 0 }));
}

/** True once at least one OQR snapshot exists (used to decide Neon vs JSON). */
export async function hasSnapshots(): Promise<boolean> {
  const db = getDb();
  if (!db) return false;
  const [snap] = await db
    .select({ id: rankingSnapshots.id })
    .from(rankingSnapshots)
    .where(and(eq(rankingSnapshots.rankingType, "oqr_men")))
    .limit(1);
  return !!snap;
}
