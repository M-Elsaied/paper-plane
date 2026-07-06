/**
 * Daily rankings sync (Vercel Cron). Fetches the OQR + WTCS + Mixed Relay
 * rankings, and only writes a NEW snapshot when the content hash changed — this
 * is how "the official ranking was updated" is detected. On a change it stores
 * the snapshot + entries and recomputes the qualification line per gender.
 *
 * Runs in seed-JSON mode as a no-op-with-note when DATABASE_URL is unset, so the
 * route is always safe to hit.
 */
import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { RANKING_IDS, type RankingKey } from "@/config/wt-api";
import { wtGet } from "@/lib/wt-api/client";
import type { RawRanking } from "@/lib/wt-api/rankings";
import { normalizeRanking } from "@/lib/wt-api/rankings";
import { computeQualificationLine } from "@/lib/engine/qualification";
import { ENGINE_VERSION } from "@/lib/engine/version";
import { getDb } from "@/db/client";
import { rankingSnapshots, rankingEntries, qualificationStates } from "@/db/schema";
import {
  authorizeCron,
  startRun,
  finishRun,
  storeRawPayload,
  rankingContentHash,
} from "@/lib/ingest/sync-run";

export const maxDuration = 300;
export const dynamic = "force-dynamic";

const TARGETS: { key: RankingKey; type: string; gender?: "male" | "female" }[] = [
  { key: "oqr_men", type: "oqr_men", gender: "male" },
  { key: "oqr_women", type: "oqr_women", gender: "female" },
  { key: "wtcs_men", type: "wtcs_men" },
  { key: "wtcs_women", type: "wtcs_women" },
  { key: "mr_olympic", type: "mr_olympic" },
];

export async function GET(req: Request) {
  if (!authorizeCron(req)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const db = getDb();
  if (!db) {
    return NextResponse.json({
      status: "noop",
      note: "seed-JSON mode (no DATABASE_URL) — provision Neon to enable persistence",
    });
  }

  const runId = await startRun("sync-rankings");
  const summary: Record<string, string> = {};

  try {
    for (const target of TARGETS) {
      const { id, expectName } = RANKING_IDS[target.key];
      const res = await wtGet<RawRanking>(`/rankings/${id}`, { limit: 1000 });
      const raw = res.data;

      // Guard: WT could renumber ids — assert the name still matches.
      if (expectName && !raw.ranking_name?.includes(expectName)) {
        summary[target.type] = `name-mismatch:${raw.ranking_name}`;
        continue;
      }

      const rawId = await storeRawPayload(runId, `/rankings/${id}`, { limit: 1000 }, raw);
      const hash = rankingContentHash(
        raw.rankings.map((r) => ({ id: r.athlete_id, rank: r.rank, total: r.total })),
      );

      const existing = await db
        .select({ id: rankingSnapshots.id })
        .from(rankingSnapshots)
        .where(and(eq(rankingSnapshots.rankingType, target.type), eq(rankingSnapshots.contentHash, hash)))
        .limit(1);
      if (existing.length) {
        summary[target.type] = "unchanged";
        continue;
      }

      const [snap] = await db
        .insert(rankingSnapshots)
        .values({
          rankingId: id,
          rankingType: target.type,
          contentHash: hash,
          publishedAt: raw.published,
          rawPayloadId: rawId ?? undefined,
        })
        .returning();

      await db.insert(rankingEntries).values(
        raw.rankings.map((r) => ({
          snapshotId: snap.id,
          athleteId: r.athlete_id,
          rank: r.rank,
          lastRank: r.last_rank,
          change: r.change != null ? String(r.change) : undefined,
          totalPoints: r.total,
        })),
      );

      // Recompute the qualification line for OQR rankings.
      if (target.gender) {
        const state = normalizeRanking(raw, target.gender);
        const line = computeQualificationLine(state.athletes);
        await db.insert(qualificationStates).values({
          snapshotId: snap.id,
          gender: target.gender,
          line,
          engineVersion: ENGINE_VERSION,
        });
      }

      summary[target.type] = `updated:${raw.rankings.length}`;
    }

    await finishRun(runId, "ok", summary);
    return NextResponse.json({ status: "ok", summary });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await finishRun(runId, "error", summary, message);
    return NextResponse.json({ status: "error", error: message, summary }, { status: 500 });
  }
}
