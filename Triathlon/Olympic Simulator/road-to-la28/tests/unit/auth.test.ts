import { describe, it, expect, vi } from "vitest";

// auth.ts imports next/headers at module load; stub it (we test pure crypto).
vi.mock("next/headers", () => ({ cookies: async () => ({ get: () => undefined, set: () => {}, delete: () => {} }) }));

import { signToken, verifyToken } from "@/lib/auth";

describe("session token crypto", () => {
  it("round-trips an account id", () => {
    const t = signToken("acc-123");
    expect(verifyToken(t)).toBe("acc-123");
  });

  it("rejects a tampered signature", () => {
    const t = signToken("acc-123");
    const [id] = t.split(".");
    expect(verifyToken(`${id}.deadbeef`)).toBeNull();
  });

  it("rejects a tampered id", () => {
    const t = signToken("acc-123");
    const [, sig] = t.split(".");
    const otherId = Buffer.from("acc-999").toString("base64url");
    expect(verifyToken(`${otherId}.${sig}`)).toBeNull();
  });

  it("rejects garbage / empty / malformed tokens without throwing", () => {
    expect(verifyToken("no-dot")).toBeNull();
    expect(verifyToken("")).toBeNull();
    expect(verifyToken(null)).toBeNull();
    expect(verifyToken(undefined)).toBeNull();
    expect(verifyToken("a.b.c.d")).toBeNull();
  });

  it("rejects a token signed with a different secret", () => {
    const orig = process.env.SESSION_SECRET;
    process.env.SESSION_SECRET = "secret-A";
    const t = signToken("acc-123");
    process.env.SESSION_SECRET = "secret-B";
    expect(verifyToken(t)).toBeNull();
    process.env.SESSION_SECRET = orig;
  });

  it("round-trips a UUID account id (the real key shape)", () => {
    const id = "3827daff-f3d9-4a80-b2bc-48a2ec0bc16d";
    expect(verifyToken(signToken(id))).toBe(id);
  });
});
