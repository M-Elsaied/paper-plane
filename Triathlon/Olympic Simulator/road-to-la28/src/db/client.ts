/**
 * Neon/Drizzle client. Lazily created so the app runs with no database at all
 * (seed-JSON mode). Cron routes call `requireDb()` which throws a clear error if
 * DATABASE_URL is unset, rather than crashing the whole app at import time.
 */
import { drizzle } from "drizzle-orm/neon-http";
import { neon } from "@neondatabase/serverless";
import * as schema from "./schema";

export type Db = ReturnType<typeof drizzle<typeof schema>>;

let _db: Db | null = null;

export function getDb(): Db | null {
  if (_db) return _db;
  const url = process.env.DATABASE_URL;
  if (!url) return null;
  _db = drizzle(neon(url), { schema });
  return _db;
}

export function requireDb(): Db {
  const db = getDb();
  if (!db) {
    throw new Error(
      "DATABASE_URL is not set. Provision Neon (vercel:marketplace) and pull env, or run in seed-JSON mode.",
    );
  }
  return db;
}

export { schema };
