/** Redeem a recovery link on a new device: verify the token, set the session, go home. */
import { NextResponse } from "next/server";
import { verifyToken, setSession } from "@/lib/auth";
import { getBoard } from "@/lib/account";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const url = new URL(req.url);
  const token = url.searchParams.get("t");
  const accountId = verifyToken(token);
  if (!accountId || !(await getBoard(accountId))) {
    return NextResponse.redirect(new URL("/account?error=invalid", url.origin));
  }
  await setSession(accountId);
  return NextResponse.redirect(new URL("/account?linked=1", url.origin));
}
