/** Send a test notification to a specific subscription (so users can confirm it works). */
import { NextResponse } from "next/server";
import { subscriptionsForEndpoint, sendToSubscription } from "@/lib/push";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { endpoint?: string };
  if (!body.endpoint) return NextResponse.json({ error: "no-endpoint" }, { status: 400 });
  const subs = await subscriptionsForEndpoint(body.endpoint);
  if (!subs.length) return NextResponse.json({ error: "not-found" }, { status: 404 });
  const sent = await sendToSubscription(subs[0], {
    title: "Road to LA28",
    body: "Alerts are on — we'll only ping you when something actually happens. 🥇",
    url: "/",
    tag: "test",
  });
  return NextResponse.json({ sent });
}
