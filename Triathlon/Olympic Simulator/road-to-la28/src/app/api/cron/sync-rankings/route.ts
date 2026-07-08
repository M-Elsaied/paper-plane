/**
 * Daily rankings sync (Vercel Cron). Fetches the OQR + WTCS + Mixed Relay
 * rankings, and only writes a NEW snapshot when the content hash changed — this
 * is how "the official ranking was updated" is detected. On a change it stores
 * athlete metadata, the snapshot + entries (with the full score breakdown), and
 * recomputes the qualification line per gender.
 *
 * Runs as a no-op-with-note when DATABASE_URL is unset (seed-JSON mode).
 */
import { NextResponse } from "next/server";
import { and, eq } from "drizzle-orm";
import { RANKING_IDS, type RankingKey } from "@/config/wt-api";
import { wtGet } from "@/lib/wt-api/client";
import type { RawRanking } from "@/lib/wt-api/rankings";
import { getDb } from "@/db/client";
import { rankingSnapshots, rankingEntries } from "@/db/schema";
import {
  authorizeCron,
  startRun,
  finishRun,
  storeRawPayload,
  rankingContentHash,
} from "@/lib/ingest/sync-run";
import { ingestOqrSnapshot, ingestMrSnapshot, ingestWorldSnapshot } from "@/lib/ingest/rankings-ingest";
import { notifyOnRankingUpdate } from "@/lib/ingest/notify";

export const maxDuration = 300;
export const dynamic = "force-dynamic";

const TARGETS: { key: RankingKey; type: string; kind: "oqr" | "world" | "wtcs" | "mr"; gender?: "male" | "female" }[] = [
  { key: "oqr_men", type: "oqr_men", kind: "oqr", gender: "male" },
  { key: "oqr_women", type: "oqr_women", kind: "oqr", gender: "female" },
  { key: "world_men", type: "world_men", kind: "world", gender: "male" },
  { key: "world_women", type: "world_women", kind: "world", gender: "female" },
  { key: "wtcs_men", type: "wtcs_men", kind: "wtcs" },
  { key: "wtcs_women", type: "wtcs_women", kind: "wtcs" },
  { key: "mr_olympic", type: "mr_olympic", kind: "mr" },
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

      if (expectName && !raw.ranking_name?.includes(expectName)) {
        summary[target.type] = `name-mismatch:${raw.ranking_name}`;
        continue;
      }

      const rawId = await storeRawPayload(runId, `/rankings/${id}`, { limit: 1000 }, raw);

      // Content hash uses athlete id or team noc depending on ranking kind.
      const rows = (raw.rankings as unknown as Record<string, unknown>[]).map((r) => ({
        id: (r.athlete_id as number) ?? (r.team_noc as string) ?? (r.rank as number),
        rank: r.rank as number,
        total: r.total as number,
      }));
      const hash = rankingContentHash(rows);

      const existing = await db
        .select({ id: rankingSnapshots.id })
        .from(rankingSnapshots)
        .where(and(eq(rankingSnapshots.rankingType, target.type), eq(rankingSnapshots.contentHash, hash)))
        .limit(1);
      if (existing.length) {
        summary[target.type] = "unchanged";
        continue;
      }

      if (target.kind === "oqr" && target.gender) {
        await ingestOqrSnapshot(db, raw, target.gender, target.type, hash, rawId);
      } else if (target.kind === "world" && target.gender) {
        await ingestWorldSnapshot(db, raw, target.gender, target.type, hash, rawId);
      } else if (target.kind === "mr") {
        await ingestMrSnapshot(db, raw as never, target.type, hash, rawId);
      } else {
        // WTCS: minimal snapshot + rank/total entries (not read by UI yet).
        const [snap] = await db
          .insert(rankingSnapshots)
          .values({ rankingId: id, rankingType: target.type, contentHash: hash, publishedAt: raw.published, rawPayloadId: rawId ?? undefined })
          .returning();
        await db.insert(rankingEntries).values(
          raw.rankings.map((r) => ({ snapshotId: snap.id, athleteId: r.athlete_id, rank: r.rank, totalPoints: r.total })),
        );
      }

      summary[target.type] = `updated:${raw.rankings.length}`;
    }

    // Event-driven alerts: notify followers about any gender whose ranking changed.
    for (const g of ["male", "female"] as const) {
      const type = g === "male" ? "oqr_men" : "oqr_women";
      if (summary[type]?.startsWith("updated")) {
        try {
          const n = await notifyOnRankingUpdate(db, g);
          summary[`notify_${g}`] = `sent:${n}`;
        } catch (e) {
          summary[`notify_${g}`] = `error:${(e as Error).message}`;
        }
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
