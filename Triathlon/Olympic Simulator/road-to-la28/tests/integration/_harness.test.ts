import { describe, it, expect } from "vitest";
import { eq } from "drizzle-orm";
import { getDb } from "@/db/client";
import { accounts } from "@/db/schema";

// Proves the PGlite + getDb-mock + migrations harness works end to end.
describe("test harness", () => {
  it("gives each test a fresh migrated PGlite via getDb()", async () => {
    const db = getDb()!;
    expect(db).not.toBeNull();
    await db.insert(accounts).values({ id: "acc-1", board: { myAthlete: null, follows: [] } });
    const [row] = await db.select().from(accounts).where(eq(accounts.id, "acc-1"));
    expect(row.id).toBe("acc-1");
  });

  it("is isolated — the previous test's row is gone", async () => {
    const db = getDb()!;
    const rows = await db.select().from(accounts);
    expect(rows.length).toBe(0);
  });
});
