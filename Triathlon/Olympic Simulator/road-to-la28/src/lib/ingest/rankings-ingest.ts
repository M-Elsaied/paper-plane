/**
 * Persist a normalized ranking snapshot to Neon: upsert athlete metadata,
 * append the snapshot + per-athlete entries (with the full score breakdown the
 * engine needs), and recompute + store the qualification line for OQR rankings.
 */
import "server-only";
import { sql } from "drizzle-orm";
import type { Db } from "@/db/client";
import { athletes, rankingSnapshots, rankingEntries, qualificationStates } from "@/db/schema";
import { computeQualificationLine } from "@/lib/engine/qualification";
import { ENGINE_VERSION } from "@/lib/engine/version";
import type { QualState } from "@/lib/engine/types";
import type { RawRanking } from "@/lib/wt-api/rankings";
import { normalizeRanking } from "@/lib/wt-api/rankings";
import type { Gender } from "@/config/pathways";

/** Upsert athlete display metadata from an OQR ranking payload. */
export async function upsertAthletes(db: Db, raw: RawRanking, gender: Gender) {
  const rows = raw.rankings.map((a) => ({
    athleteId: a.athlete_id,
    fullName: a.athlete_full_name,
    noc: a.athlete_noc,
    gender: a.athlete_gender ?? gender,
    yearOfBirth: a.athlete_yob,
    headshotUrl: a.athlete_profile_image ?? undefined,
    flagUrl: a.athlete_flag_circle ?? undefined,
  }));
  if (!rows.length) return;
  await db
    .insert(athletes)
    .values(rows)
    .onConflictDoUpdate({
      target: athletes.athleteId,
      set: {
        fullName: sql`excluded.full_name`,
        noc: sql`excluded.noc`,
        gender: sql`excluded.gender`,
        yearOfBirth: sql`excluded.year_of_birth`,
        headshotUrl: sql`excluded.headshot_url`,
        flagUrl: sql`excluded.flag_url`,
        updatedAt: sql`now()`,
      },
    });
}

/** Insert an OQR snapshot: entries carry the full Score[] so the engine can
 *  rebuild state; also recompute + store the qualification line. */
export async function ingestOqrSnapshot(
  db: Db,
  raw: RawRanking,
  gender: Gender,
  rankingType: string,
  contentHash: string,
  rawPayloadId: number | null,
): Promise<QualState> {
  const state = normalizeRanking(raw, gender);
  await upsertAthletes(db, raw, gender);

  const [snap] = await db
    .insert(rankingSnapshots)
    .values({
      rankingId: raw.ranking_id,
      rankingType,
      contentHash,
      publishedAt: raw.published,
      rawPayloadId: rawPayloadId ?? undefined,
    })
    .returning();

  const byId = new Map(state.athletes.map((a) => [a.athleteId, a]));
  await db.insert(rankingEntries).values(
    raw.rankings.map((r) => ({
      snapshotId: snap.id,
      athleteId: r.athlete_id,
      rank: r.rank,
      lastRank: r.last_rank,
      change: r.change != null ? String(r.change) : undefined,
      totalPoints: r.total,
      scores: byId.get(r.athlete_id)?.scores ?? [],
    })),
  );

  const line = computeQualificationLine(state.athletes);
  await db.insert(qualificationStates).values({
    snapshotId: snap.id,
    gender,
    line,
    engineVersion: ENGINE_VERSION,
  });

  return state;
}

/** Insert a Mixed Relay snapshot (nations, not athletes). */
export async function ingestMrSnapshot(
  db: Db,
  raw: { ranking_id: number; published: string; rankings: { team_noc?: string; team_country_name?: string; team_title?: string; rank: number; total: number }[] },
  rankingType: string,
  contentHash: string,
  rawPayloadId: number | null,
) {
  const [snap] = await db
    .insert(rankingSnapshots)
    .values({
      rankingId: raw.ranking_id,
      rankingType,
      contentHash,
      publishedAt: raw.published,
      rawPayloadId: rawPayloadId ?? undefined,
    })
    .returning();

  await db.insert(rankingEntries).values(
    raw.rankings.map((r) => ({
      snapshotId: snap.id,
      nationNoc: r.team_noc || r.team_country_name || r.team_title || "—",
      rank: r.rank,
      totalPoints: r.total,
    })),
  );
}
