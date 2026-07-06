/**
 * Passwordless session for the account bridge. Identity is an HMAC-signed token
 * carrying the account id, stored in an httpOnly cookie. A recovery link is the
 * same signed token in a URL — the user's self-held "magic link" (no email
 * service needed for v1). Not high-security (no PII, no payments) — it guards a
 * synced list of followed athletes.
 */
import "server-only";
import { createHmac, timingSafeEqual } from "node:crypto";
import { cookies } from "next/headers";

const COOKIE = "rtla28_session";
const MAX_AGE = 60 * 60 * 24 * 365; // 1 year

function secret(): string {
  return process.env.SESSION_SECRET || "dev-insecure-secret-set-SESSION_SECRET";
}

function b64url(buf: Buffer | string): string {
  return Buffer.from(buf).toString("base64url");
}

/** token = base64url(accountId).base64url(hmac(accountId)) */
export function signToken(accountId: string): string {
  const sig = createHmac("sha256", secret()).update(accountId).digest();
  return `${b64url(accountId)}.${b64url(sig)}`;
}

export function verifyToken(token: string | undefined | null): string | null {
  if (!token || !token.includes(".")) return null;
  const [idPart, sigPart] = token.split(".");
  let accountId: string;
  try {
    accountId = Buffer.from(idPart, "base64url").toString();
  } catch {
    return null;
  }
  const expected = createHmac("sha256", secret()).update(accountId).digest();
  let given: Buffer;
  try {
    given = Buffer.from(sigPart, "base64url");
  } catch {
    return null;
  }
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) return null;
  return accountId;
}

export async function setSession(accountId: string) {
  const store = await cookies();
  store.set(COOKIE, signToken(accountId), {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: MAX_AGE,
  });
}

export async function getSessionAccountId(): Promise<string | null> {
  const store = await cookies();
  return verifyToken(store.get(COOKIE)?.value);
}

export async function clearSession() {
  const store = await cookies();
  store.delete(COOKIE);
}
