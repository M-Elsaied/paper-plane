/** A fresh in-memory Postgres (PGlite) migrated from ./drizzle, per call.
 *  Hermetic + fast — no Neon, no network. The `getDb()` seam is mocked to
 *  return this instance (see setup.ts). */
import { PGlite } from "@electric-sql/pglite";
import { drizzle } from "drizzle-orm/pglite";
import { migrate } from "drizzle-orm/pglite/migrator";
import * as schema from "@/db/schema";

export type TestDb = ReturnType<typeof drizzle<typeof schema>>;

export async function makeTestDb(): Promise<TestDb> {
  const client = new PGlite();
  const db = drizzle(client, { schema });
  await migrate(db, { migrationsFolder: "./drizzle" });
  return db;
}
