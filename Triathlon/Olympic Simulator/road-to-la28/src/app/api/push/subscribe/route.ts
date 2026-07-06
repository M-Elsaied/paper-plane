/** Store a browser push subscription + the athletes the user follows (for targeting). */
import { NextResponse } from "next/server";
import { getSessionAccountId } from "@/lib/auth";
import { saveSubscription, deleteSubscription, type BrowserSubscription } from "@/lib/push";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as {
    subscription?: BrowserSubscription;
    follows?: number[];
  };
  if (!body.subscription?.endpoint) {
    return NextResponse.json({ error: "no-subscription" }, { status: 400 });
  }
  const accountId = await getSessionAccountId();
  const ok = await saveSubscription(accountId, body.subscription, body.follows ?? []);
  if (!ok) return NextResponse.json({ error: "no-database" }, { status: 503 });
  return NextResponse.json({ ok: true });
}

export async function DELETE(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { endpoint?: string };
  if (body.endpoint) await deleteSubscription(body.endpoint);
  return NextResponse.json({ ok: true });
}
