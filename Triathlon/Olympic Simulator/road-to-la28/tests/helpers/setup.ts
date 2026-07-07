/**
 * Integration setup: mock the DB seam with a fresh PGlite per test, and the
 * next/headers cookie store with an in-memory jar. Imported via
 * vitest.workspace.ts setupFiles for the `integration` project.
 */
import { vi, beforeEach, afterEach } from "vitest";
import { makeTestDb } from "./db";
import { makeCookieJar, type CookieJar } from "./cookies";

// Holders are hoisted so the vi.mock factories (also hoisted) can close over them.
const holder = vi.hoisted(() => ({ db: null as unknown, cookies: null as unknown }));

vi.mock("@/db/client", async () => {
  const actual = await vi.importActual<typeof import("@/db/client")>("@/db/client");
  return { ...actual, getDb: () => holder.db, requireDb: () => holder.db };
});

vi.mock("next/headers", () => ({
  cookies: async () => holder.cookies,
}));

beforeEach(async () => {
  holder.db = await makeTestDb();
  holder.cookies = makeCookieJar();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** The current test's PGlite-backed drizzle instance (for seeding fixtures). */
export function testDb() {
  return holder.db as Awaited<ReturnType<typeof makeTestDb>>;
}

/** The current cookie jar (to simulate a browser holding a session). */
export function cookieJar(): CookieJar {
  return holder.cookies as CookieJar;
}

/** Simulate a "fresh device" — drop all cookies mid-test. */
export function resetCookies() {
  holder.cookies = makeCookieJar();
}

/** Override the mocked db (e.g. set null to exercise the no-database path). */
export function setTestDb(db: unknown) {
  holder.db = db;
}
