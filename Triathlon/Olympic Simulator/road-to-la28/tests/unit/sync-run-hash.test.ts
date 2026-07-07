import { describe, it, expect, afterEach, vi } from "vitest";
import { rankingContentHash, authorizeCron } from "@/lib/ingest/sync-run";

describe("rankingContentHash", () => {
  const rows = [
    { id: 1, rank: 1, total: 100 },
    { id: 2, rank: 2, total: 90 },
    { id: 3, rank: 3, total: 80 },
  ];

  it("is order-invariant (shuffled rows → same hash)", () => {
    const shuffled = [rows[2], rows[0], rows[1]];
    expect(rankingContentHash(shuffled)).toBe(rankingContentHash(rows));
  });

  it("changes when any total changes", () => {
    const changed = [{ ...rows[0], total: 101 }, rows[1], rows[2]];
    expect(rankingContentHash(changed)).not.toBe(rankingContentHash(rows));
  });

  it("changes when a rank changes", () => {
    const changed = [{ ...rows[0], rank: 4 }, rows[1], rows[2]];
    expect(rankingContentHash(changed)).not.toBe(rankingContentHash(rows));
  });
});

describe("authorizeCron", () => {
  afterEach(() => vi.unstubAllEnvs());

  const req = (auth?: string) => new Request("http://t/api/cron", { headers: auth ? { authorization: auth } : {} });

  it("allows when no secret is set and not production", () => {
    vi.stubEnv("NODE_ENV", "test");
    vi.stubEnv("CRON_SECRET", "");
    expect(authorizeCron(req())).toBe(true);
  });

  it("blocks in production when no secret is set", () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("CRON_SECRET", "");
    expect(authorizeCron(req())).toBe(false);
  });

  it("requires the exact bearer when a secret is set", () => {
    vi.stubEnv("CRON_SECRET", "abc");
    expect(authorizeCron(req())).toBe(false);
    expect(authorizeCron(req("Bearer wrong"))).toBe(false);
    expect(authorizeCron(req("Bearer abc"))).toBe(true);
  });
});
