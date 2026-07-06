/**
 * Shared cron plumbing: auth guard, sync-run bookkeeping, content hashing.
 */
import "server-only";
import { createHash } from "node:crypto";
import { getDb } from "@/db/client";
import { syncRuns, rawPayloads } from "@/db/schema";

/** Verify the Vercel Cron / manual trigger is authorized. */
export function authorizeCron(req: Request): boolean {
  const secret = process.env.CRON_SECRET;
  if (!secret) return process.env.NODE_ENV !== "production"; // dev: allow
  const auth = req.headers.get("authorization");
  return auth === `Bearer ${secret}`;
}

/** Deterministic content hash of a ranking's ordered (athlete, rank, total) tuples. */
export function rankingContentHash(
  rows: { id: number | string; rank: number; total: number }[],
): string {
  const canonical = rows
    .map((r) => `${r.id}:${r.rank}:${r.total}`)
    .sort()
    .join("|");
  return createHash("sha256").update(canonical).digest("hex");
}

export async function startRun(job: string): Promise<number | null> {
  const db = getDb();
  if (!db) return null;
  const [row] = await db.insert(syncRuns).values({ job, status: "running" }).returning();
  return row.id;
}

export async function finishRun(
  id: number | null,
  status: "ok" | "error" | "noop",
  stats?: unknown,
  error?: string,
) {
  const db = getDb();
  if (!db || id == null) return;
  await db
    .update(syncRuns)
    .set({ status, finishedAt: new Date(), stats: stats ?? null, error: error ?? null })
    .where(eqId(id));
}

export async function storeRawPayload(
  syncRunId: number | null,
  endpoint: string,
  params: unknown,
  payload: unknown,
): Promise<number | null> {
  const db = getDb();
  if (!db) return null;
  const payloadHash = createHash("sha256").update(JSON.stringify(payload)).digest("hex");
  const [row] = await db
    .insert(rawPayloads)
    .values({ syncRunId: syncRunId ?? undefined, endpoint, params, payload, payloadHash })
    .returning();
  return row.id;
}

import { eq } from "drizzle-orm";
function eqId(id: number) {
  return eq(syncRuns.id, id);
}
