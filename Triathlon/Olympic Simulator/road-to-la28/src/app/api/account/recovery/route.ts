/** Returns the user's self-held recovery link (signed token) for the claimed account. */
import { NextResponse } from "next/server";
import { getSessionAccountId, signToken } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const id = await getSessionAccountId();
  if (!id) return NextResponse.json({ error: "not-claimed" }, { status: 401 });
  const origin = new URL(req.url).origin;
  const url = `${origin}/claim?t=${encodeURIComponent(signToken(id))}`;
  return NextResponse.json({ url });
}
