/**
 * Clear ranking data so the next sync re-ingests with the current schema.
 * Usage:  npx tsx scripts/reset-rankings.ts
 */
import { config } from "dotenv";
config({ path: ".env.local" });

import { getDb } from "../src/db/client";
import { qualificationStates, rankingEntries, rankingSnapshots, athletes } from "../src/db/schema";

async function main() {
  const db = getDb();
  if (!db) throw new Error("no DATABASE_URL");
  await db.delete(qualificationStates);
  await db.delete(rankingEntries);
  await db.delete(rankingSnapshots);
  await db.delete(athletes);
  console.log("✓ cleared qualificationStates, rankingEntries, rankingSnapshots, athletes");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
