/**
 * Pick-'Em data access — submit/read a user's podium pick, aggregate the crowd
 * forecast, and score picks once results are official.
 */
import "server-only";
import { and, eq, isNull, sql } from "drizzle-orm";
import type { Gender } from "@/config/pathways";
import { getDb } from "@/db/client";
import { picks } from "@/db/schema";
import { scorePick } from "@/lib/engine/pickem";

export interface CrowdForecast {
  total: number;
  /** Per athlete: how many picked them to win / to make the podium. */
  athletes: { athleteId: number; win: number; podium: number }[];
}

export async function submitPick(
  ownerKey: string,
  raceId: number,
  gender: Gender,
  podium: number[],
): Promise<boolean> {
  const db = getDb();
  if (!db) return false;
  await db
    .insert(picks)
    .values({ ownerKey, raceId, gender, podium })
    .onConflictDoUpdate({
      target: [picks.raceId, picks.gender, picks.ownerKey],
      set: { podium, score: null, perfect: null, scoredAt: null },
    });
  return true;
}

export async function getMyPick(ownerKey: string, raceId: number, gender: Gender) {
  const db = getDb();
  if (!db) return null;
  const [row] = await db
    .select()
    .from(picks)
    .where(and(eq(picks.raceId, raceId), eq(picks.gender, gender), eq(picks.ownerKey, ownerKey)))
    .limit(1);
  return row ?? null;
}

export async function getCrowd(raceId: number, gender: Gender): Promise<CrowdForecast> {
  const db = getDb();
  if (!db) return { total: 0, athletes: [] };
  const rows = await db
    .select({ podium: picks.podium })
    .from(picks)
    .where(and(eq(picks.raceId, raceId), eq(picks.gender, gender)));

  const win = new Map<number, number>();
  const pod = new Map<number, number>();
  for (const r of rows) {
    const p = (r.podium as number[]) ?? [];
    if (p[0] != null) win.set(p[0], (win.get(p[0]) ?? 0) + 1);
    p.forEach((id) => pod.set(id, (pod.get(id) ?? 0) + 1));
  }
  const athletes = [...pod.keys()]
    .map((athleteId) => ({ athleteId, win: win.get(athleteId) ?? 0, podium: pod.get(athleteId) ?? 0 }))
    .sort((a, b) => b.win - a.win || b.podium - a.podium);

  return { total: rows.length, athletes };
}

/** Distinct (raceId, gender) that still have unscored picks. */
export async function racesNeedingScoring(): Promise<{ raceId: number; gender: Gender }[]> {
  const db = getDb();
  if (!db) return [];
  const rows = await db
    .selectDistinct({ raceId: picks.raceId, gender: picks.gender })
    .from(picks)
    .where(isNull(picks.score));
  return rows.map((r) => ({ raceId: r.raceId, gender: r.gender as Gender }));
}

/** Score every unscored pick for a race against the actual podium. Returns count. */
export async function scoreRace(
  raceId: number,
  gender: Gender,
  actualPodium: number[],
): Promise<{ scored: number; owners: string[] }> {
  const db = getDb();
  if (!db || actualPodium.length < 3) return { scored: 0, owners: [] };
  const rows = await db
    .select()
    .from(picks)
    .where(and(eq(picks.raceId, raceId), eq(picks.gender, gender), isNull(picks.score)));

  const owners: string[] = [];
  for (const row of rows) {
    const s = scorePick((row.podium as number[]) ?? [], actualPodium);
    await db
      .update(picks)
      .set({ score: s.score, perfect: s.perfect, scoredAt: new Date() })
      .where(eq(picks.id, row.id));
    owners.push(row.ownerKey);
  }
  return { scored: rows.length, owners };
}

/** A user's overall Pick-'Em record (for a streak/accuracy stat). */
export async function ownerRecord(ownerKey: string) {
  const db = getDb();
  if (!db) return { picks: 0, totalScore: 0, perfects: 0 };
  const [row] = await db
    .select({
      n: sql<number>`count(*)`,
      total: sql<number>`coalesce(sum(${picks.score}), 0)`,
      perfects: sql<number>`coalesce(sum(case when ${picks.perfect} then 1 else 0 end), 0)`,
    })
    .from(picks)
    .where(and(eq(picks.ownerKey, ownerKey), sql`${picks.score} is not null`));
  return { picks: Number(row?.n ?? 0), totalScore: Number(row?.total ?? 0), perfects: Number(row?.perfects ?? 0) };
}
